"""
Configuration for Generational Knowledge Distillation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenKDConfig:
    """Configuration for the generational KD pipeline.

    Attributes:
        model_names: List of HuggingFace model names/paths.
                     model_names[0] = teacher, model_names[1..N] = students.
        common_dim:  Dimension of the shared projection space.
        learning_rate: Learning rate for each student generation.
        batch_size:  Training batch size.
        max_seq_len: Maximum sequence length for tokenization.
        epochs:      Number of epochs per generation.
        pooling_mode: Pooling strategy — "mean" or "attention".
        weight_strategy: How to compute loss weights — "uniform" or "linear_decay".
        device:      Device string, e.g. "cuda" or "cpu".
        checkpoint_dir: Directory to save per-generation checkpoints.
        dataset_name: HuggingFace dataset name for training data.
        dataset_split: Which split to use (e.g. "train").
        dataset_text_field: Name of the text column in the dataset.
        max_samples: Optional cap on dataset size (useful for testing).
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_steps: Number of warmup steps for the LR scheduler.
        log_every:   Log training metrics every N steps.
        seed:        Random seed for reproducibility.
    """

    # --- Models ---
    model_names: List[str] = field(default_factory=lambda: [
        "sshleifer/tiny-gpt2",   # Teacher (M[0])
        "sshleifer/tiny-gpt2",   # Student 1 (M[1])
        "sshleifer/tiny-gpt2",   # Student 2 (M[2])
    ])

    # --- Projection ---
    common_dim: int = 256

    # --- Training ---
    learning_rate: float = 5e-5
    batch_size: int = 8
    max_seq_len: int = 128
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    seed: int = 42

    # --- Pooling & Loss ---
    pooling_mode: str = "mean"           # "mean" | "attention"
    weight_strategy: str = "uniform"     # "uniform" | "linear_decay"

    # --- Dataset ---
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    max_samples: Optional[int] = None

    # --- I/O ---
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50

    @property
    def num_generations(self) -> int:
        """Number of student generations (excludes the teacher)."""
        return len(self.model_names) - 1

    def get_loss_weights(self, generation_k: int) -> List[float]:
        """Return the loss weights w[k][0..k-1] for student generation k.

        Args:
            generation_k: The 1-based generation index of the current student.

        Returns:
            A list of floats of length k, one weight per predecessor.
        """
        if self.weight_strategy == "uniform":
            return [1.0 / generation_k] * generation_k

        elif self.weight_strategy == "linear_decay":
            # More recent predecessors get higher weight.
            # w[i] = (i+1) / sum(1..k)
            total = sum(range(1, generation_k + 1))
            return [(i + 1) / total for i in range(generation_k)]

        else:
            raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")
