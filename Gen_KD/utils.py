"""
Utility helpers for Generational Knowledge Distillation.
"""

import logging
import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------- #
#  Logging
# ---------------------------------------------------------------------- #

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger."""
    logger = logging.getLogger("Gen_KD")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------- #
#  Reproducibility
# ---------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Set random seed across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- #
#  Checkpointing
# ---------------------------------------------------------------------- #

def save_checkpoint(
    model: torch.nn.Module,
    projection: torch.nn.Module,
    generation: int,
    checkpoint_dir: str,
) -> str:
    """Save model + projection checkpoint for a given generation.

    Args:
        model:          The student model wrapper.
        projection:     The corresponding projection head.
        generation:     1-based generation index.
        checkpoint_dir: Base directory for checkpoints.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"gen_{generation}.pt")

    model_state_dict = model.model.state_dict() if hasattr(model, "model") else model.state_dict()
    projection_state_dict = projection.state_dict()
    torch.save(
        {
            "generation": generation,
            "model_name": getattr(model, "model_name", None),
            "student_state_dict": model_state_dict,
            "model_state_dict": model_state_dict,
            "proj_student_state_dict": projection_state_dict,
            "projection_state_dict": projection_state_dict,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    projection: torch.nn.Module,
) -> int:
    """Load a checkpoint into model + projection.

    Returns:
        The generation index stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu")
    model_state_dict = ckpt["model_state_dict"]
    if hasattr(model, "model"):
        model.model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    projection.load_state_dict(ckpt["projection_state_dict"])
    return ckpt["generation"]
