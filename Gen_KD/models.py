"""
Model wrapper for loading HuggingFace models, checkpoints, and hidden states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        self.checkpoint_path: Optional[str] = None
        self.loaded_checkpoint: Optional[dict[str, Any]] = None

        if self._looks_like_checkpoint(model_name_or_path):
            self._load_from_checkpoint(Path(model_name_or_path))
        else:
            self._load_from_pretrained(model_name_or_path)

        self.model.to(device)

        # Expose hidden dimension
        self.hidden_dim = self.model.config.hidden_size

    def _looks_like_checkpoint(self, model_name_or_path: str) -> bool:
        path = Path(model_name_or_path)
        return path.suffix == ".pt" and path.exists()

    def _load_from_pretrained(self, model_name_or_path: str) -> None:
        self.model_name = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    def _resolve_base_model_name(self, checkpoint: dict[str, Any]) -> str:
        config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
        candidates = [
            checkpoint.get("model_name"),
            checkpoint.get("base_model_name"),
            config.get("student_model_id"),
            config.get("teacher_model_id"),
            config.get("model_name"),
            config.get("model_id"),
        ]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        raise KeyError(
            "Could not resolve a base model name from checkpoint. "
            "Add 'model_name' or 'config.student_model_id' to the checkpoint."
        )

    def _extract_state_dict(self, checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
        for key in (
            "model_state_dict",
            "student_state_dict",
            "state_dict",
        ):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        raise KeyError("Checkpoint does not contain a model state dict")

    def _load_from_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

        self.loaded_checkpoint = checkpoint
        self.checkpoint_path = str(checkpoint_path)

        base_model_name = self._resolve_base_model_name(checkpoint)
        self.model_name = base_model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        missing, unexpected = self.model.load_state_dict(
            self._extract_state_dict(checkpoint),
            strict=False,
        )
        if missing or unexpected:
            print(
                f"Warning: checkpoint load for {checkpoint_path} had "
                f"missing={len(missing)} unexpected={len(unexpected)} keys"
            )

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        """Run the model and return HuggingFace causal LM outputs.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask:  (batch, seq_len)
            labels:         Optional labels for causal LM loss.

        Returns:
            HuggingFace causal LM output object.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=False,
        )

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
