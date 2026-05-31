"""
Generational Knowledge Distillation Trainer.

Implements the core training loop from the GenKD pseudo code:

    For k in 1..N:                         # each student generation
        Freeze M[0..k-1]
        For each batch x:
            H_prev[i] = M[i](x)  for i in 0..k-1   (no grad)
            H_k       = M[k](x)                      (grad)
            p_prev[i] = pool(P[i](H_prev[i]))
            p_k       = pool(P[k](H_k))
            loss       = Σ  w[k][i] · MSE(p_k, p_prev[i])
            loss.backward()
            update(M[k], P[k])
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from Gen_KD.config import GenKDConfig
from Gen_KD.models import ModelWrapper
from Gen_KD.projection import ProjectionHead, pool
from Gen_KD.utils import setup_logging, save_checkpoint


class GenKDTrainer:
    """Orchestrates multi-generation distillation."""

    def __init__(
        self,
        config: GenKDConfig,
        models: list[ModelWrapper],
        dataloader: DataLoader,
    ):
        """
        Args:
            config:     GenKDConfig instance.
            models:     List of ModelWrapper instances — models[0] is the
                        teacher, models[1..N] are students.
            dataloader: DataLoader yielding dicts with 'input_ids' and
                        'attention_mask'.
        """
        self.config = config
        self.models = models
        self.dataloader = dataloader
        self.logger = setup_logging()

        # Build one projection head per model
        self.projections: list[ProjectionHead] = []
        for m in self.models:
            proj = ProjectionHead(m.hidden_dim, config.common_dim).to(config.device)
            self.projections.append(proj)

        # MSE loss (reduction='mean')
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------------ #
    #  Main entry
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """Run the full generational distillation pipeline."""

        # Freeze the teacher (M[0])
        self.models[0].freeze()
        self.logger.info(f"Teacher frozen: {self.models[0]}")

        for k in range(1, len(self.models)):
            self._train_generation(k)

        self.logger.info("All generations trained ✓")

    # ------------------------------------------------------------------ #
    #  Per-generation training
    # ------------------------------------------------------------------ #
    def _train_generation(self, k: int) -> None:
        """Train student generation k, distilling from models 0..k-1."""

        cfg = self.config
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"  Generation {k} / {cfg.num_generations}")
        self.logger.info(f"{'='*60}")

        # Ensure all predecessors are frozen
        for i in range(k):
            self.models[i].freeze()
            self.projections[i].eval()
            for p in self.projections[i].parameters():
                p.requires_grad = False

        # Unfreeze current student + its projection
        self.models[k].unfreeze()
        self.projections[k].train()
        for p in self.projections[k].parameters():
            p.requires_grad = True

        # Optimizer for student model + its projection head
        trainable_params = (
            list(self.models[k].parameters())
            + list(self.projections[k].parameters())
        )
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

        # Simple linear warmup scheduler
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=cfg.warmup_steps,
        )

        # Loss weights for this generation
        weights = cfg.get_loss_weights(k)
        self.logger.info(f"  Loss weights w[{k}] = {weights}")

        global_step = 0
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.dataloader:
                loss = self._train_step(k, batch, weights)

                # Gradient accumulation
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps
                num_batches += 1
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = epoch_loss / num_batches
                    self.logger.info(
                        f"  [Gen {k}] Epoch {epoch+1}/{cfg.epochs}  "
                        f"Step {global_step}  Loss {avg:.6f}"
                    )

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.logger.info(
                f"  [Gen {k}] Epoch {epoch+1} complete — "
                f"Avg loss: {avg_epoch_loss:.6f}"
            )

        # Freeze the now-trained student and save checkpoint
        self.models[k].freeze()
        ckpt_path = save_checkpoint(
            self.models[k], self.projections[k], k, cfg.checkpoint_dir
        )
        self.logger.info(f"  Checkpoint saved → {ckpt_path}")

    # ------------------------------------------------------------------ #
    #  Single training step
    # ------------------------------------------------------------------ #
    def _train_step(
        self,
        k: int,
        batch: dict,
        weights: list[float],
    ) -> torch.Tensor:
        """Compute the generational distillation loss for one batch.

        Matches the pseudo code:
            loss = Σ_{i=0}^{k-1}  w[k][i] · MSE(p_k, p_prev[i])
        """
        device = self.config.device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # ---- Predecessors (no grad) ---- #
        p_prev = []
        with torch.no_grad():
            for i in range(k):
                h_i = self.models[i](input_ids, attention_mask)     # (B, S, D_i)
                z_i = self.projections[i](h_i)                      # (B, S, common_dim)
                p_i = pool(z_i, attention_mask, self.config.pooling_mode)  # (B, common_dim)
                p_prev.append(p_i)

        # ---- Current student (with grad) ---- #
        h_k = self.models[k](input_ids, attention_mask)
        z_k = self.projections[k](h_k)
        p_k = pool(z_k, attention_mask, self.config.pooling_mode)

        # ---- Weighted MSE loss ---- #
        total_loss = torch.tensor(0.0, device=device)
        for i in range(k):
            total_loss = total_loss + weights[i] * self.mse(p_k, p_prev[i])

        return total_loss
