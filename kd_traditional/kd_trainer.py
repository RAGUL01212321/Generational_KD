"""
Knowledge Distillation Trainer.
Implements the distillation pipeline from pseudocode.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import json
import math
import re
import logging
from datetime import datetime

from huggingface_hub import snapshot_download
from transformers import Adafactor, AutoTokenizer, AutoModelForCausalLM
from kd_config import KDConfig
from kd_projection import TeacherProjection, StudentProjection
from kd_data import create_dataloader
from kd_monitoring import TrainingMonitor, LossVisualizer
from kd_monitoring import TrainingMonitor, LossVisualizer

import matplotlib.pyplot as plt


class KDTrainer:
    """Knowledge Distillation trainer."""
    
    def __init__(self, config: KDConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._resolved_model_paths = {}
        
        # Create checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("="*70)
        self.logger.info("KNOWLEDGE DISTILLATION TRAINING START")
        self.logger.info("="*70)
        self._save_run_metadata()
        
        # Load models
        print("Loading teacher model (frozen)...")
        self.logger.info("Loading teacher model (frozen)...")
        self.teacher = self._load_model(
            config.teacher_model_id, 
            freeze=config.teacher_frozen
        )
        
        print("Loading student model (trainable)...")
        self.logger.info("Loading student model (trainable)...")
        self.student = self._load_model(config.student_model_id, freeze=False)
        
        # Get dimensions
        teacher_dim = self.teacher.config.hidden_size
        student_dim = self.student.config.hidden_size
        common_dim = config.common_dim
        
        print(f"\nModel dimensions:")
        print(f"  Teacher hidden: {teacher_dim}")
        print(f"  Student hidden: {student_dim}")
        print(f"  Common dim: {common_dim}\n")
        
        # Create projection layers
        self.proj_teacher = TeacherProjection(teacher_dim, common_dim).to(self.device).float()
        self.proj_student = StudentProjection(student_dim, common_dim).to(self.device).float()
        self.proj_teacher.eval()
        for param in self.proj_teacher.parameters():
            param.requires_grad = False
        
        # Tokenizer
        self.tokenizer = self._load_tokenizer(config.student_model_id)
        
        # Loss functions
        self.loss_fn = nn.MSELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Optimizer (only for student and projection layers).
        # Adafactor keeps much less optimizer state than AdamW, which is
        # important when the teacher, student, and trainable state share one GPU.
        trainable_params = (
            list(self.student.parameters()) +
            list(self.proj_student.parameters())
        )
        self.optimizer = Adafactor(
            trainable_params,
            lr=config.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
        )
        
        # Training state
        self.global_step = 0
        self.train_losses = []
        self.last_kd_loss = 0.0
        self.last_ce_loss = 0.0
        self._trainable_params = trainable_params
        
        # Initialize monitor
        self.monitor = TrainingMonitor(config.checkpoint_dir)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup file and console logging."""
        logger = logging.getLogger("KDTrainer")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create logs directory
        log_dir = Path(self.config.checkpoint_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File handler for all logs
        log_file = log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # File handler for loss logs
        loss_file = log_dir / f"losses_{timestamp}.log"
        loss_handler = logging.FileHandler(loss_file)
        loss_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        loss_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(loss_handler)
        
        return logger

    def _save_run_metadata(self):
        """Save run configuration next to checkpoints and logs."""
        metadata_path = Path(self.config.checkpoint_dir) / "run_config.json"
        metadata = {
            "started_at": datetime.now().isoformat(),
            "checkpoint_dir": self.config.checkpoint_dir,
            "config": self.config.__dict__,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _safe_snapshot_path(self, model_id: str) -> Path:
        """Return a local model path whose basename is safe for dynamic imports."""
        if model_id in self._resolved_model_paths:
            return self._resolved_model_paths[model_id]

        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", model_id).strip("_")
        local_dir = Path(".hf_model_snapshots") / safe_name

        print(f"Downloading/caching {model_id} to {local_dir} for custom code loading...")
        snapshot_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
        )

        resolved_path = Path(snapshot_path)
        self._resolved_model_paths[model_id] = resolved_path
        return resolved_path

    def _load_tokenizer(self, model_id: str):
        """Load tokenizer, using a safe local path when custom code is needed."""
        try:
            return AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=False,
            )
        except Exception as e:
            print(f"Warning: Native tokenizer loading failed ({e}), trying with custom code...")
            model_path = self._safe_snapshot_path(model_id)
            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

    def _load_model(self, model_id: str, freeze: bool = False):
        """Load model from HuggingFace."""
        # Keep frozen teacher weights small, but train the student in fp32.
        # AdamW on fp16 model parameters is a common source of NaNs without AMP
        # gradient scaling.
        torch_dtype = torch.float16 if freeze and self.config.use_fp16 else torch.float32

        try:
            # Try to load with native transformers code (no custom modeling)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=False,
                torch_dtype=torch_dtype,
            )
        except Exception as e:
            print(f"Warning: Native loading failed ({e}), trying with custom code...")
            # Transformers 4.37 can build invalid module paths for repo ids
            # containing dots, such as Apollo-1.8B. A local safe basename avoids
            # imports like transformers_modules.FreedomIntelligence.Apollo-1.
            model_path = self._safe_snapshot_path(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        model.to(self.device)
        
        if freeze:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            model.train()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            # Enable gradient checkpointing for student model to reduce memory
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        
        return model
    
    def _get_backbone(self, model):
        """Return the transformer backbone so KD does not compute LM logits."""
        for attr_name in ("model", "transformer", "base_model"):
            backbone = getattr(model, attr_name, None)
            if backbone is not None and backbone is not model:
                return backbone
        return model

    def _extract_hidden_states(self, model, inputs):
        """Extract only final hidden states, avoiding logits and all-layer states."""
        backbone = self._get_backbone(model)
        outputs = backbone(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    def _compute_student_logits(self, hidden_states):
        """Project student hidden states through the LM head for CE loss."""
        lm_head = getattr(self.student, "lm_head", None)
        if lm_head is None:
            raise AttributeError(
                "Student model does not expose an lm_head; cannot compute CE loss "
                "without running the full causal LM forward."
            )
        return lm_head(hidden_states)

    def _causal_lm_loss(self, logits, input_ids, attention_mask):
        """Next-token prediction loss, ignoring padding tokens."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
        return self.ce_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
           
    def _pool_hidden_states(self, hidden_states, attention_mask):
        """Mean-pool hidden states over non-padding tokens only."""
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        return (hidden_states * mask).sum(dim=1) / token_counts

    def _has_non_finite(self, name: str, tensor: torch.Tensor) -> bool:
        """Log and detect NaN/Inf tensors before they corrupt model weights."""
        if torch.isfinite(tensor).all():
            return False

        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() > 0:
            min_val = finite.min().item()
            max_val = finite.max().item()
            print(f"Warning: non-finite values in {name}; finite range [{min_val:.4e}, {max_val:.4e}]")
        else:
            print(f"Warning: all values are non-finite in {name}")
        return True
    
    def train_step(self, batch, loss_scale: int = 1) -> float:
        """
        Single training step following the pseudocode:
        
        for each batch x:
            with no_grad:
                H_T = T(x)
            H_S = S(x)
            
            Z_T = P_T(H_T)
            Z_S = P_S(H_S)
            
            p_T = mean(Z_T)
            p_S = mean(Z_S)
            
            kd_loss = MSE(p_S, p_T)
            ce_loss = CrossEntropy(student_logits, shifted_input_ids)
            loss = kd_weight * kd_loss + ce_weight * ce_loss
            loss.backward()
            update(S) every gradient_accumulation_steps microbatches
            update(P_S) every gradient_accumulation_steps microbatches
        """
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # ===== Forward Pass =====
        
        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            H_T = self._extract_hidden_states(self.teacher, inputs)
            # Convert to float32 for projection compatibility
            H_T = H_T.float()
            Z_T = self.proj_teacher(H_T)  # (batch, seq_len, common_dim)
            p_T = self._pool_hidden_states(Z_T, attention_mask)  # (batch, common_dim)
        
        # Student forward (trainable)
        H_S = self._extract_hidden_states(self.student, inputs)
        # Convert to float32 for projection compatibility
        H_S = H_S.float()
        
        # ===== Projection =====
        
        Z_S = self.proj_student(H_S)  # (batch, seq_len, common_dim)
        
        # ===== Pooling =====
        
        p_S = self._pool_hidden_states(Z_S, attention_mask)  # (batch, common_dim)
        
        # ===== Loss =====
        
        if (
            self._has_non_finite("teacher pooled states", p_T)
            or self._has_non_finite("student pooled states", p_S)
        ):
            self.optimizer.zero_grad(set_to_none=True)
            return float("nan")

        kd_loss = self.loss_fn(p_S, p_T)
        logits = self._compute_student_logits(H_S)
        ce_loss = self._causal_lm_loss(logits, input_ids, attention_mask)
        loss = (
            self.config.kd_loss_weight * kd_loss
            + self.config.ce_loss_weight * ce_loss
        )
        if not torch.isfinite(loss):
            print("Warning: non-finite loss before backward; skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
            return float("nan")
        
        # ===== Backprop =====
        
        raw_loss = loss.detach().item()
        self.last_kd_loss = kd_loss.detach().item()
        self.last_ce_loss = ce_loss.detach().item()
        (loss / loss_scale).backward()
        return raw_loss

    def _optimizer_step(self) -> bool:
        """Clip gradients and update trainable parameters after accumulation."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._trainable_params,
            self.config.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            print("Warning: non-finite gradients; skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
            return False

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Clear GPU cache to reduce memory fragmentation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return True
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        min_batch_loss = float('inf')
        halfway_checkpoint_saved = False
        accumulation_steps = max(1, self.config.gradient_accumulation_steps)

        total_batches = len(dataloader)
        if self.config.max_steps:
            remaining_steps = max(0, self.config.max_steps - self.global_step)
            total_batches = min(total_batches, remaining_steps)
        halfway_point = total_batches // 2

        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= total_batches:
                break

            accumulation_window_start = (batch_idx // accumulation_steps) * accumulation_steps
            accumulation_window_end = min(
                accumulation_window_start + accumulation_steps,
                total_batches,
            )
            accumulation_window_size = accumulation_window_end - accumulation_window_start

            loss = self.train_step(batch, loss_scale=accumulation_window_size)
            self.global_step += 1

            if math.isfinite(loss):
                total_loss += loss
                num_batches += 1
                min_batch_loss = min(min_batch_loss, loss)
            else:
                skipped_batches += 1
                print(f"Step {self.global_step:5d} | skipped non-finite batch")
                continue

            should_step_optimizer = (
                (batch_idx + 1) >= accumulation_window_end
                or (self.config.max_steps and self.global_step >= self.config.max_steps)
            )
            if should_step_optimizer and not self._optimizer_step():
                skipped_batches += 1
                continue
            
            # Log to monitor
            self.monitor.log_step(self.global_step, loss)
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                avg_loss = total_loss / max(num_batches, 1)
                log_msg = (
                    f"Step {self.global_step:5d} | "
                    f"Loss: {loss:.4f} | "
                    f"KD: {self.last_kd_loss:.4f} | "
                    f"CE: {self.last_ce_loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
                print(log_msg)
                self.logger.info(log_msg)
            
            # Checkpoint at halfway point
            if batch_idx >= halfway_point and not halfway_checkpoint_saved:
                self.save_checkpoint("halfway")
                halfway_checkpoint_saved = True
            
            # Max steps check
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        if num_batches == 0:
            raise RuntimeError(
                "Every batch in this epoch produced non-finite values. "
                "Restart from a clean checkpoint and try a lower learning rate."
            )

        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        if skipped_batches:
            print(f"Skipped {skipped_batches} non-finite batches this epoch.")
        
        # End epoch tracking
        self.monitor.end_epoch(avg_epoch_loss, min_batch_loss)
        
        return avg_epoch_loss
    
    def save_checkpoint(self, name: str = "latest"):
        """Save checkpoint with atomic write to temp file."""
        import tempfile
        import shutil
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        tmp_path = None
        
        try:
            # Move models to CPU for saving to avoid GPU memory issues
            student_on_device = next(self.student.parameters()).device
            proj_student_on_device = next(self.proj_student.parameters()).device
            proj_teacher_on_device = next(self.proj_teacher.parameters()).device
            
            self.student.cpu()
            self.proj_student.cpu()
            self.proj_teacher.cpu()
            
            # Create checkpoint data on CPU
            checkpoint_data = {
                "global_step": self.global_step,
                "student_state_dict": self.student.state_dict(),
                "proj_student_state_dict": self.proj_student.state_dict(),
                "proj_teacher_state_dict": self.proj_teacher.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "train_losses": self.train_losses,
            }
            
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(
                dir=self.config.checkpoint_dir, 
                suffix='.pt', 
                delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
            
            torch.save(checkpoint_data, tmp_path)
            
            # Atomic move to final destination
            shutil.move(tmp_path, str(checkpoint_path))
            
            save_msg = f"✓ Saved checkpoint: {checkpoint_path}"
            print(save_msg)
            self.logger.info(save_msg)
            
            # Move models back to device
            self.student.to(student_on_device)
            self.proj_student.to(proj_student_on_device)
            self.proj_teacher.to(proj_teacher_on_device)
            
        except Exception as e:
            error_msg = f"⚠ Warning: Failed to save checkpoint {name}: {e}"
            print(error_msg)
            self.logger.warning(error_msg)
            # Clean up temp file if it exists
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except:
                    pass
            # Ensure models are back on device
            try:
                self.student.to(student_on_device)
                self.proj_teacher.to(proj_teacher_on_device)
                self.proj_student.to(proj_student_on_device)
            except:
                pass
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.proj_student.load_state_dict(checkpoint["proj_student_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(
                "Warning: optimizer state could not be loaded. "
                f"Continuing with a fresh {self.config.optimizer} optimizer. ({e})"
            )
        self.global_step = checkpoint["global_step"]
        self.train_losses = checkpoint["train_losses"]
        
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # Start monitoring
        self.monitor.start_training()
        
        # Create dataloader
        dataloader, dataset = create_dataloader(
            dataset_path=self.config.dataset_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            max_samples=self.config.max_samples,
        )
        
        print(f"Training for {num_epochs} epochs\n")
        print("=" * 60)
        
        # Log training config
        self.logger.info(f"Training Configuration:")
        self.logger.info(f"  Epochs: {num_epochs}")
        self.logger.info(f"  Batch Size: {self.config.batch_size}")
        self.logger.info(f"  Gradient Accumulation Steps: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Learning Rate: {self.config.learning_rate}")
        self.logger.info(f"  Dataset Size: {len(dataset)} samples")
        self.logger.info(f"  Teacher Model: {self.config.teacher_model_id}")
        self.logger.info(f"  Student Model: {self.config.student_model_id}")
        self.logger.info(f"  Device: {self.config.device}")
        
        try:
            for epoch in range(num_epochs):
                self.monitor.start_epoch(epoch)
                
                epoch_msg = f"\nEpoch {epoch + 1}/{num_epochs}"
                print(epoch_msg)
                print("-" * 60)
                self.logger.info(epoch_msg)
                
                try:
                    avg_loss = self.train_epoch(dataloader)
                    epoch_complete_msg = f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}"
                    print(epoch_complete_msg)
                    self.logger.info(epoch_complete_msg)
                    
                    # Save checkpoint after successful epoch
                    self.save_checkpoint(f"epoch_{epoch + 1}")
                except Exception as e:
                    error_msg = f"Error in Epoch {epoch + 1}:\n{str(e)}\n\nTraceback:\n"
                    import traceback as tb
                    error_msg += tb.format_exc()
                    self.monitor.log_error(error_msg)
                    self.logger.error(error_msg)
                    print(f"✗ Error in epoch {epoch + 1}: {e}")
                    raise
            
            print("\n" + "=" * 60)
            print("Training complete!")
            self.logger.info("Training complete!")
            
        except KeyboardInterrupt:
            interrupt_msg = "\n\n⚠ Training interrupted by user"
            print(interrupt_msg)
            self.logger.warning(interrupt_msg)
        except Exception as e:
            error_msg = f"Critical Error During Training:\n{str(e)}\n\nTraceback:\n"
            import traceback as tb
            error_msg += tb.format_exc()
            self.monitor.log_error(error_msg)
            self.logger.error(error_msg)
            raise
        finally:
            # Always save final checkpoint and metrics
            self.save_checkpoint("final")
            
            # Print monitoring summary
            self.monitor.print_summary()
            
            # Save all metrics
            self._save_metrics()

        plt.ioff()
        plt.show()

    
    def _save_metrics(self):
        """Save training metrics."""
        # Save via monitor (comprehensive)
        self.monitor.save_metrics("metrics_detailed.json")
        self.monitor.save_csv("training_log.csv")
        self.monitor.save_training_log()
        
        # Legacy format (for compatibility)
        metrics = {
            "train_losses": self.train_losses,
            "global_step": self.global_step,
            "timestamp": datetime.now().isoformat(),
        }
        
        metrics_path = Path(self.config.checkpoint_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved metrics: {metrics_path}")
