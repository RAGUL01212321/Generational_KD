"""
Validation Configuration
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    
    # Dataset
    validation_dataset_size: int = 1000  # Number of validation samples
    validation_dataset_file: str = "Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa_2000_val.json"
    training_dataset_file: str = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json"
    
    # Models
    teacher_model_id: str = "Qwen/Qwen1.5-1.8B"
    student_model_id: str = "Qwen/Qwen1.5-0.5B"
    
    # Validation
    batch_size: int = 4
    seq_length: int = 512
    device: str = "cuda"
    
    # Checkpoints to validate
    base_student_checkpoint: Optional[str] = None  # If None, load fresh model
    distilled_student_checkpoint: Optional[str] = "kd_checkpoints/final.pt"
    
    # Projection
    common_dim: int = 768
    
    # Paths
    validation_dir: str = "kd_validation"
    results_file: str = "kd_validation/results/validation_results.json"
    
    # Qualitative evaluation
    num_qualitative_samples: int = 20
    generation_max_length: int = 150
    generation_temperature: float = 0.7
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
