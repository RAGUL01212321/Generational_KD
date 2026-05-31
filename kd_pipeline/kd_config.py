"""
Knowledge Distillation Configuration
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class KDConfig:
    """
    Knowledge Distillation settings.
    """
    
    # Models
    teacher_model_id: str = "Qwen/Qwen1.5-1.8B"
    student_model_id: str = "Qwen/Qwen1.5-0.5B"
    
    # Projection layer
    common_dim: int = 768  # Common hidden dimension
    
    # Training
    batch_size: int = 2
    learning_rate: float = 1e-5
    optimizer: str = "adafactor"
    kd_loss_weight: float = 0.6
    ce_loss_weight: float = 0.4
    num_epochs: int = 3
    warmup_steps: int = 500
    max_steps: Optional[int] = None
    
    # Gradient
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Data
    dataset_path: str = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json"
    dataset_file: Optional[str] = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json"
    max_samples: Optional[int] = None  # None = use all
    seq_length: int = 512
    
    # Checkpointing
    checkpoint_dir: str = "kd_checkpoints"
    save_every: int = 500  # steps (reduced I/O frequency)
    
    # Logging
    log_every: int = 50  # steps (reduced I/O frequency)
    
    # Device
    device: str = "cuda"
    
    # Teacher
    teacher_frozen: bool = True
    
    # Precision
    use_fp16: bool = True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
