"""
Generational Knowledge Distillation Trainer.

Matches the teacher / assistant / student pseudocode used for the single-step
and generational distillation flow:

    teacher.eval()
    assistant.eval()
    freeze teacher + assistant
    for each batch:
        H_T = teacher(x, output_hidden_states=True).hidden_states[-1]
        H_A = assistant(x, output_hidden_states=True).hidden_states[-1]
        out_S = student(x, labels=labels, output_hidden_states=True)
        H_S = out_S.hidden_states[-1]
        Z_T = proj_teacher(H_T)
        Z_A = proj_assistant(H_A)
        Z_S = proj_student(H_S)
        p_T = mean_pool(Z_T)
        p_A = mean_pool(Z_A)
        p_S = mean_pool(Z_S)
        loss_kd = (MSE(p_S, p_T) + MSE(p_S, p_A)) / 2
        loss = 0.6 * loss_kd + 0.4 * out_S.loss
        loss.backward()
        optimizer.step()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Adafactor

from Gen_KD.config import GenKDConfig
from Gen_KD.models import ModelWrapper
from Gen_KD.projection import ProjectionHead, pool
from Gen_KD.utils import save_checkpoint, setup_logging


class GenKDTrainer:
    """Orchestrates teacher + assistant + student distillation."""

    def __init__(
        self,
        config: GenKDConfig,
        models: list[ModelWrapper],
        dataloader: DataLoader,
    ):
        self.config = config
        self.models = models
        self.dataloader = dataloader
        self.logger = setup_logging()

        if len(self.models) < 3:
            raise ValueError(
                "Gen_KD expects at least a teacher, an assistant, and a student."
            )

        self._validate_hidden_dims()

        # One projection head per model.
        self.projections: list[ProjectionHead] = []
        for model in self.models:
            projection = ProjectionHead(model.hidden_dim, config.common_dim).to(config.device)

            # If the model came from a checkpoint, reuse its saved projection weights
            # when available. This keeps the assistant target mathematically aligned
            # with the checkpoint that produced it.
            if getattr(model, "loaded_checkpoint", None):
                checkpoint = model.loaded_checkpoint or {}
                projection_state = (
                    checkpoint.get("projection_state_dict")
                    or checkpoint.get("proj_student_state_dict")
                    or checkpoint.get("proj_teacher_state_dict")
                )
                if isinstance(projection_state, dict):
                    try:
                        self._load_projection_state(
                            projection,
                            projection_state,
                            model.model_name,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            f"Could not load projection checkpoint for {model.model_name}: {exc}"
                        )

            self.projections.append(projection)

        self.mse = nn.MSELoss()
        self.optimizer: Optional[Adafactor] = None

    def _validate_hidden_dims(self) -> None:
        """Verify model hidden sizes before creating projection heads."""
        expected = self.config.expected_hidden_dims
        if expected is None:
            return
        if len(expected) != len(self.models):
            raise ValueError(
                "expected_hidden_dims must match model_names length "
                f"({len(expected)} != {len(self.models)})."
            )

        actual = [model.hidden_dim for model in self.models]
        mismatches = [
            (idx, exp, got)
            for idx, (exp, got) in enumerate(zip(expected, actual))
            if exp != got
        ]
        if mismatches:
            details = ", ".join(
                f"model[{idx}] expected {exp}, got {got}"
                for idx, exp, got in mismatches
            )
            raise ValueError(f"Hidden dimension mismatch: {details}")

    def _load_projection_state(
        self,
        projection: ProjectionHead,
        projection_state: dict[str, torch.Tensor],
        model_name: str,
    ) -> None:
        """Load projection weights from either Gen_KD or kd_pipeline checkpoints."""
        normalized_state = {}
        for key, value in projection_state.items():
            if key.startswith("projection.linear."):
                normalized_key = key.replace("projection.linear.", "linear.", 1)
            else:
                normalized_key = key
            normalized_state[normalized_key] = value

        incompatible = projection.load_state_dict(normalized_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            self.logger.warning(
                "Projection checkpoint for %s had missing=%s unexpected=%s",
                model_name,
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

    def train(self) -> None:
        """Run the full generational distillation pipeline."""
        self.models[0].freeze()
        self.projections[0].eval()
        for param in self.projections[0].parameters():
            param.requires_grad = False

        self.logger.info(f"Teacher frozen: {self.models[0]}")

        for student_idx in range(2, len(self.models)):
            self._train_generation(student_idx)

        self.logger.info("All generations trained ✓")

    def _train_generation(self, student_idx: int) -> None:
        """Train one student generation using the teacher and its assistant."""
        cfg = self.config
        assistant_idx = student_idx - 1

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"  Student model index {student_idx}")
        self.logger.info(f"{'=' * 60}")

        # Freeze teacher and assistant.
        for idx in (0, assistant_idx):
            self.models[idx].freeze()
            self.projections[idx].eval()
            for param in self.projections[idx].parameters():
                param.requires_grad = False

        # Train only the current student and its projection.
        self.models[student_idx].unfreeze()
        self.projections[student_idx].train()
        for param in self.projections[student_idx].parameters():
            param.requires_grad = True

        trainable_params = list(self.models[student_idx].parameters()) + list(
            self.projections[student_idx].parameters()
        )
        self.optimizer = Adafactor(
            trainable_params,
            lr=cfg.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
        )

        self.logger.info(
            "  Loss weights = kd:{:.3f} ce:{:.3f}".format(
                cfg.kd_loss_weight,
                cfg.ce_loss_weight,
            )
        )

        global_step = 0
        accumulation_steps = max(1, cfg.gradient_accumulation_steps)

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch in self.dataloader:
                loss = self._train_step(0, assistant_idx, student_idx, batch)
                if not torch.isfinite(loss):
                    self.logger.warning(
                        f"  [Gen {student_idx}] Skipping non-finite batch at step {global_step + 1}"
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / accumulation_steps).backward()
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if global_step % cfg.log_every == 0:
                    avg = epoch_loss / max(num_batches, 1)
                    self.logger.info(
                        f"  [Gen {student_idx}] Epoch {epoch + 1}/{cfg.epochs} "
                        f"Step {global_step} Loss {avg:.6f}"
                    )

            if num_batches == 0:
                raise RuntimeError(
                    f"Generation {student_idx} produced no finite training batches."
                )

            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(
                f"  [Gen {student_idx}] Epoch {epoch + 1} complete — Avg loss: {avg_epoch_loss:.6f}"
            )

        self.models[student_idx].freeze()
        checkpoint_path = save_checkpoint(
            self.models[student_idx],
            self.projections[student_idx],
            student_idx,
            cfg.checkpoint_dir,
        )
        self.logger.info(f"  Checkpoint saved → {checkpoint_path}")

    def _train_step(
        self,
        teacher_idx: int,
        assistant_idx: int,
        student_idx: int,
        batch: dict,
    ) -> torch.Tensor:
        """Compute the exact teacher/assistant/student loss for one batch."""
        device = self.config.device
        teacher_input_ids = batch[f"input_ids_{teacher_idx}"].to(device)
        teacher_attention_mask = batch[f"attention_mask_{teacher_idx}"].to(device)
        assistant_input_ids = batch[f"input_ids_{assistant_idx}"].to(device)
        assistant_attention_mask = batch[f"attention_mask_{assistant_idx}"].to(device)
        student_input_ids = batch[f"input_ids_{student_idx}"].to(device)
        student_attention_mask = batch[f"attention_mask_{student_idx}"].to(device)

        # Build causal-LM labels from the input batch.
        labels = student_input_ids.clone()
        labels = labels.masked_fill(student_attention_mask == 0, -100)

        with torch.no_grad():
            teacher_out = self.models[teacher_idx](
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            assistant_out = self.models[assistant_idx](
                input_ids=assistant_input_ids,
                attention_mask=assistant_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            h_teacher = teacher_out.hidden_states[-1]
            h_assistant = assistant_out.hidden_states[-1]

            z_teacher = self.projections[teacher_idx](h_teacher)
            z_assistant = self.projections[assistant_idx](h_assistant)

            p_teacher = pool(z_teacher, teacher_attention_mask, self.config.pooling_mode)
            p_assistant = pool(
                z_assistant,
                assistant_attention_mask,
                self.config.pooling_mode,
            )

        student_out = self.models[student_idx](
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        h_student = student_out.hidden_states[-1]
        z_student = self.projections[student_idx](h_student)
        p_student = pool(z_student, student_attention_mask, self.config.pooling_mode)

        kd_teacher = self.mse(p_student, p_teacher)
        kd_assistant = self.mse(p_student, p_assistant)
        loss_kd = (kd_teacher + kd_assistant) / 2

        loss_ce = student_out.loss
        if loss_ce is None:
            loss_ce = torch.tensor(0.0, device=device)

        return self.config.kd_loss_weight * loss_kd + self.config.ce_loss_weight * loss_ce
