"""
Gen_KD — Generational Knowledge Distillation

Sequential multi-generation distillation where each student learns
from all preceding models (teacher + earlier students).
"""

from Gen_KD.config import GenKDConfig
from Gen_KD.models import ModelWrapper
from Gen_KD.projection import ProjectionHead, pool
from Gen_KD.trainer import GenKDTrainer

__all__ = ["GenKDConfig", "ModelWrapper", "ProjectionHead", "pool", "GenKDTrainer"]
