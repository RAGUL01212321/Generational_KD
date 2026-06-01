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

import csv
import json
import math
import time
from pathlib import Path
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
        self.global_step = 0
        self.total_optimizer_updates = 0
        self.training_start_time = 0.0
        self.last_step_metrics: dict[str, torch.Tensor | float | int] = {}
        self.train_metric_rows: list[dict[str, float | int]] = []
        self.gradient_metric_rows: list[dict[str, float | int]] = []

        self.run_output_dir = self._next_run_output_dir()
        self.metrics_log_dir = self.run_output_dir / config.metrics_log_dir
        self.plots_dir = self.run_output_dir / config.plots_dir
        self.train_metrics_path = self.metrics_log_dir / "train_metrics.csv"
        self.gradient_metrics_path = self.metrics_log_dir / "gradient_metrics.csv"
        self.summary_path = self.metrics_log_dir / "training_summary.json"
        self._initialize_metric_outputs()

        self.checkpoint_dir = self.run_output_dir / config.checkpoint_dir
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _next_run_output_dir(self) -> Path:
        """Return the next GenN_run directory for this training launch."""
        run_root = Path(self.config.run_output_root)
        generation_name = f"Gen{len(self.models) - 1}"
        run_root.mkdir(parents=True, exist_ok=True)

        max_run = 0
        for path in run_root.iterdir():
            if not path.is_dir() or not path.name.startswith(f"{generation_name}_"):
                continue
            suffix = path.name[len(generation_name) + 1:]
            if suffix.isdigit():
                max_run = max(max_run, int(suffix))

        return run_root / f"{generation_name}_{max_run + 1}"

    def _initialize_metric_outputs(self) -> None:
        """Create CSV files with headers before training starts."""
        self.run_output_dir.mkdir(parents=True, exist_ok=False)
        self.metrics_log_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Run outputs  : {self.run_output_dir}")
        self.logger.info(f"Metrics CSV  : {self.train_metrics_path}")
        self.logger.info(f"Gradient CSV : {self.gradient_metrics_path}")
        self.logger.info(f"Plots dir    : {self.plots_dir}")

        with self.train_metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "epoch",
                    "total_loss",
                    "kd_loss",
                    "kd_teacher",
                    "kd_assistant",
                    "ce_loss",
                    "learning_rate",
                ]
            )

        with self.gradient_metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "projection_grad_norm",
                    "first_layer_grad_norm",
                    "middle_layer_grad_norm",
                    "last_layer_grad_norm",
                ]
            )

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

    def _tensor_to_float(self, value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu().item())
        return float(value)

    def _current_learning_rate(self) -> float:
        if self.optimizer is None or not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _write_train_metrics(self, epoch: int) -> None:
        """Append one scalar metrics row. Called only at logging intervals."""
        if not self.last_step_metrics:
            return

        row = {
            "step": self.global_step,
            "epoch": epoch,
            "total_loss": self._tensor_to_float(self.last_step_metrics["total_loss"]),
            "kd_loss": self._tensor_to_float(self.last_step_metrics["kd_loss"]),
            "kd_teacher": self._tensor_to_float(self.last_step_metrics["kd_teacher"]),
            "kd_assistant": self._tensor_to_float(self.last_step_metrics["kd_assistant"]),
            "ce_loss": self._tensor_to_float(self.last_step_metrics["ce_loss"]),
            "learning_rate": self._current_learning_rate(),
        }
        self.train_metric_rows.append(row)

        with self.train_metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _parameter_grad_norm(self, parameters) -> float:
        squared_norm = 0.0
        found_grad = False
        for param in parameters:
            if param.grad is None:
                continue
            found_grad = True
            grad = param.grad.detach().float()
            squared_norm += float(torch.sum(grad * grad).cpu().item())
        if not found_grad:
            return 0.0
        return math.sqrt(squared_norm)

    def _get_transformer_layers(self, student_idx: int) -> list[nn.Module]:
        model = self.models[student_idx].model
        candidates = [
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("backbone", "layers"),
        ]
        for first_attr, second_attr in candidates:
            parent = getattr(model, first_attr, None)
            layers = getattr(parent, second_attr, None) if parent is not None else None
            if layers is not None and len(layers) > 0:
                return list(layers)

        layers = getattr(model, "layers", None)
        if layers is not None and len(layers) > 0:
            return list(layers)
        return []

    def _write_gradient_metrics(self, student_idx: int) -> None:
        """Append lightweight gradient norms using existing gradients only."""
        projection_grad_norm = self._parameter_grad_norm(
            self.projections[student_idx].parameters()
        )

        layers = self._get_transformer_layers(student_idx)
        if layers:
            first_layer = layers[0]
            middle_layer = layers[len(layers) // 2]
            last_layer = layers[-1]
            first_norm = self._parameter_grad_norm(first_layer.parameters())
            middle_norm = self._parameter_grad_norm(middle_layer.parameters())
            last_norm = self._parameter_grad_norm(last_layer.parameters())
        else:
            first_norm = middle_norm = last_norm = 0.0

        row = {
            "step": self.global_step,
            "projection_grad_norm": projection_grad_norm,
            "first_layer_grad_norm": first_norm,
            "middle_layer_grad_norm": middle_norm,
            "last_layer_grad_norm": last_norm,
        }
        self.gradient_metric_rows.append(row)

        with self.gradient_metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _checkpoint_metadata(self, epoch: int) -> dict[str, float | int]:
        if not self.last_step_metrics:
            return {"step": self.global_step, "epoch": epoch}
        return {
            "step": self.global_step,
            "epoch": epoch,
            "total_loss": self._tensor_to_float(self.last_step_metrics["total_loss"]),
            "kd_loss": self._tensor_to_float(self.last_step_metrics["kd_loss"]),
            "ce_loss": self._tensor_to_float(self.last_step_metrics["ce_loss"]),
            "kd_teacher": self._tensor_to_float(self.last_step_metrics["kd_teacher"]),
            "kd_assistant": self._tensor_to_float(self.last_step_metrics["kd_assistant"]),
        }

    def _write_training_summary(self) -> None:
        elapsed = time.perf_counter() - self.training_start_time
        final_metrics = self._checkpoint_metadata(epoch=self.config.epochs)

        if self.train_metric_rows:
            min_total_loss = min(row["total_loss"] for row in self.train_metric_rows)
            min_kd_loss = min(row["kd_loss"] for row in self.train_metric_rows)
            min_ce_loss = min(row["ce_loss"] for row in self.train_metric_rows)
        else:
            min_total_loss = final_metrics.get("total_loss", 0.0)
            min_kd_loss = final_metrics.get("kd_loss", 0.0)
            min_ce_loss = final_metrics.get("ce_loss", 0.0)

        summary = {
            "total_training_steps": self.global_step,
            "total_optimizer_updates": self.total_optimizer_updates,
            "total_epochs": self.config.epochs,
            "final_total_loss": final_metrics.get("total_loss", 0.0),
            "final_kd_loss": final_metrics.get("kd_loss", 0.0),
            "final_ce_loss": final_metrics.get("ce_loss", 0.0),
            "minimum_total_loss": min_total_loss,
            "minimum_kd_loss": min_kd_loss,
            "minimum_ce_loss": min_ce_loss,
            "total_training_time_seconds": elapsed,
        }

        with self.summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

    def _plot_metric(
        self,
        rows: list[dict[str, float | int]],
        y_keys: list[str],
        title: str,
        ylabel: str,
        output_name: str,
    ) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        if rows:
            steps = [row["step"] for row in rows]
            for y_key in y_keys:
                ax.plot(steps, [row[y_key] for row in rows], label=y_key, linewidth=1.8)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No logged data", ha="center", va="center")

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(self.plots_dir / output_name)
        plt.close(fig)

    def _generate_plots(self) -> None:
        self._plot_metric(
            self.train_metric_rows,
            ["total_loss"],
            "Total Loss vs Step",
            "Total Loss",
            "total_loss_vs_step.png",
        )
        self._plot_metric(
            self.train_metric_rows,
            ["kd_loss"],
            "KD Loss vs Step",
            "KD Loss",
            "kd_loss_vs_step.png",
        )
        self._plot_metric(
            self.train_metric_rows,
            ["ce_loss"],
            "CE Loss vs Step",
            "CE Loss",
            "ce_loss_vs_step.png",
        )
        self._plot_metric(
            self.train_metric_rows,
            ["kd_teacher"],
            "Teacher KD Loss vs Step",
            "Teacher KD Loss",
            "kd_teacher_vs_step.png",
        )
        self._plot_metric(
            self.train_metric_rows,
            ["kd_assistant"],
            "Assistant KD Loss vs Step",
            "Assistant KD Loss",
            "kd_assistant_vs_step.png",
        )
        self._plot_metric(
            self.train_metric_rows,
            ["kd_loss", "ce_loss"],
            "KD vs CE Loss Comparison",
            "Loss",
            "kd_vs_ce_comparison.png",
        )
        self._plot_metric(
            self.gradient_metric_rows,
            ["projection_grad_norm"],
            "Projection Gradient Norm",
            "Gradient Norm",
            "projection_grad_norm.png",
        )
        self._plot_metric(
            self.gradient_metric_rows,
            [
                "first_layer_grad_norm",
                "middle_layer_grad_norm",
                "last_layer_grad_norm",
            ],
            "Transformer Layer Gradient Norms",
            "Gradient Norm",
            "transformer_grad_norms.png",
        )

    def train(self) -> None:
        """Run the full generational distillation pipeline."""
        self.training_start_time = time.perf_counter()
        self.models[0].freeze()
        self.projections[0].eval()
        for param in self.projections[0].parameters():
            param.requires_grad = False

        self.logger.info(f"Teacher frozen: {self.models[0]}")

        for student_idx in range(2, len(self.models)):
            self._train_generation(student_idx)

        self._write_training_summary()
        self._generate_plots()
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

        accumulation_steps = max(1, cfg.gradient_accumulation_steps)
        last_epoch = 0
        steps_per_epoch = len(self.dataloader)
        half_epoch_step = max(1, steps_per_epoch // 2)

        for epoch in range(cfg.epochs):
            last_epoch = epoch + 1
            epoch_loss = 0.0
            num_batches = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(self.dataloader):
                loss = self._train_step(0, assistant_idx, student_idx, batch)
                if not torch.isfinite(loss):
                    self.logger.warning(
                        f"  [Gen {student_idx}] Skipping non-finite batch at step {self.global_step + 1}"
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / accumulation_steps).backward()
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                if cfg.gradient_log_every > 0 and self.global_step % cfg.gradient_log_every == 0:
                    self._write_gradient_metrics(student_idx)

                if self.global_step % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.total_optimizer_updates += 1

                if self.global_step % cfg.log_every == 0:
                    self._write_train_metrics(epoch + 1)
                    avg = epoch_loss / max(num_batches, 1)
                    kd_loss = self._tensor_to_float(self.last_step_metrics["kd_loss"])
                    ce_loss = self._tensor_to_float(self.last_step_metrics["ce_loss"])
                    kd_teacher = self._tensor_to_float(
                        self.last_step_metrics["kd_teacher"]
                    )
                    kd_assistant = self._tensor_to_float(
                        self.last_step_metrics["kd_assistant"]
                    )
                    self.logger.info(
                        f"  [Gen {student_idx}] Epoch {epoch + 1}/{cfg.epochs} "
                        f"Step {self.global_step} AvgLoss {avg:.6f} "
                        f"KD {kd_loss:.6f} CE {ce_loss:.6f} "
                        f"KD_T {kd_teacher:.6f} KD_A {kd_assistant:.6f}"
                    )

                # --------------------------------------------------
                # Half-epoch checkpoint
                # --------------------------------------------------

                current_step_in_epoch = batch_idx + 1

                if current_step_in_epoch == half_epoch_step:

                    checkpoint_path = save_checkpoint(
                        self.models[student_idx],
                        self.projections[student_idx],
                        student_idx,
                        str(self.checkpoint_dir),
                        metadata=self._checkpoint_metadata(epoch + 1),
                        suffix=f"epoch{epoch+1}_half",
                    )

                    self.logger.info(
                        f"  [Gen {student_idx}] Half-epoch checkpoint saved -> {checkpoint_path}"
                    )

            if num_batches == 0:
                raise RuntimeError(
                    f"Generation {student_idx} produced no finite training batches."
                )

            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(
                f"  [Gen {student_idx}] Epoch {epoch + 1} complete — Avg loss: {avg_epoch_loss:.6f}"
            )

            # --------------------------------------------------
            # Full-epoch checkpoint
            # --------------------------------------------------
            checkpoint_path = save_checkpoint(
                self.models[student_idx],
                self.projections[student_idx],
                student_idx,
                str(self.checkpoint_dir),
                metadata=self._checkpoint_metadata(epoch + 1),
                suffix=f"epoch{epoch+1}_full",
            )
            self.logger.info(
                f"  [Gen {student_idx}] Full-epoch checkpoint saved -> {checkpoint_path}"
            )

        self.models[student_idx].freeze()
        checkpoint_path = save_checkpoint(
            self.models[student_idx],
            self.projections[student_idx],
            student_idx,
            str(self.checkpoint_dir),
            metadata=self._checkpoint_metadata(last_epoch),
            suffix="final",
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

        total_loss = self.config.kd_loss_weight * loss_kd + self.config.ce_loss_weight * loss_ce
        self.last_step_metrics = {
            "total_loss": total_loss.detach(),
            "kd_loss": loss_kd.detach(),
            "kd_teacher": kd_teacher.detach(),
            "kd_assistant": kd_assistant.detach(),
            "ce_loss": loss_ce.detach(),
        }

        return total_loss
