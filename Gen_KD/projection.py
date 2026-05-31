"""
Projection heads and pooling utilities for Generational KD.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection to a shared common dimension.

    Projects hidden states from a model's native hidden_dim to a
    shared `common_dim` so that representations from different-sized
    models can be compared via MSE.

    Architecture:  Linear → LayerNorm → GELU → Linear
    """

    def __init__(self, input_dim: int, common_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, common_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, common_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------- #
#  Pooling
# ---------------------------------------------------------------------- #

def pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Pool token-level representations to a single sequence-level vector.

    Args:
        hidden:          (batch, seq_len, dim)
        attention_mask:  (batch, seq_len)  — 1 for real tokens, 0 for padding
        mode:            "mean" (default) or "cls"

    Returns:
        pooled: (batch, dim)
    """
    if mode == "mean":
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()       # (B, S, 1)
        summed = (hidden * mask).sum(dim=1)               # (B, D)
        lengths = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        return summed / lengths

    elif mode == "cls":
        # Use the first token representation
        return hidden[:, 0, :]

    else:
        raise ValueError(f"Unknown pooling mode: {mode!r}")
