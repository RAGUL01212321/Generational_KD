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
                     model_names[0] = teacher, model_names[1] = assistant,
                     model_names[2] = student.
        expected_hidden_dims: Expected hidden sizes for teacher, assistant,
                     and student. Set to None to disable validation.
        common_dim:  Dimension of the shared projection space.
        learning_rate: Learning rate for each student generation.
        batch_size:  Training batch size.
        max_seq_len: Maximum sequence length for tokenization.
        epochs:      Number of epochs per generation.
        pooling_mode: Pooling strategy: "mean" or "cls".
        device:      Device string, e.g. "cuda" or "cpu".
        checkpoint_dir: Directory to save per-generation checkpoints.
        dataset_path: Local JSON dataset path, relative to the repo root if using a file.
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
        "Qwen/Qwen1.5-1.8B",  # Teacher (M[0])
        "./kd_pipeline/kd_checkpoints/Qwen_3/final.pt",  # Assistant (M[1]) (repo-relative)
        "HuggingFaceTB/SmolLM2-360M",  # Student (M[2])
    ])
    expected_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [
        2048,  # Qwen teacher
        1024,  # Distilled Qwen assistant
        960,   # SmolLM2-360M student
    ])

    # --- Projection ---
    common_dim: int = 768

    # --- Training ---
    optimizer: str = "adafactor"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    batch_size: int = 2
    max_seq_len: int = 512
    epochs: int = 3
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    seed: int = 42

    # --- Pooling & Loss ---
    pooling_mode: str = "mean"           # "mean" | "cls"
    kd_loss_weight: float = 0.6
    ce_loss_weight: float = 0.4

    # --- Dataset ---
    dataset_path: str = "./Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    max_samples: Optional[int] = None

    # --- I/O ---
    device: str = "cuda"
    checkpoint_dir: str = "kd_checkpoints"
    log_every: int = 50

    @property
    def num_generations(self) -> int:
        """Number of trainable student generations (teacher + assistant excluded)."""
        return max(0, len(self.model_names) - 2)
