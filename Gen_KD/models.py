"""
Model wrapper for loading HuggingFace models and extracting hidden states.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ModelWrapper(nn.Module):
    """Thin wrapper around a HuggingFace model.

    Loads a pretrained model and exposes a forward method that returns the
    last hidden state.  Provides freeze/unfreeze helpers for the
    generational distillation loop.
    """

    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name_or_path
        self.device = device

        # Load model
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model.to(device)

        # Expose hidden dimension
        self.hidden_dim = self.model.config.hidden_size

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model and return the last hidden state.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask:  (batch, seq_len)

        Returns:
            hidden_states:  (batch, seq_len, hidden_dim)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs.last_hidden_state  # (B, S, D)

    # --------------------------------------------------------------------- #
    #  Freeze / unfreeze helpers
    # --------------------------------------------------------------------- #
    def freeze(self):
        """Freeze all parameters (used for teacher + trained students)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        """Unfreeze all parameters (used for the current student)."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def __repr__(self) -> str:
        return (
            f"ModelWrapper(name={self.model_name!r}, "
            f"hidden_dim={self.hidden_dim}, "
            f"frozen={not next(self.model.parameters()).requires_grad})"
        )


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """Load and configure a tokenizer for the given model.

    Sets the pad token to eos_token if the tokenizer doesn't have one.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
