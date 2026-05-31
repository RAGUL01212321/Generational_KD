"""
Projection layers for KD.
Projects teacher and student hidden states to common dimension.
"""

import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """Simple projection layer."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, output_dim)
        """
        return self.linear(x)


class TeacherProjection(nn.Module):
    """Teacher projection layer (frozen during training)."""
    
    def __init__(self, teacher_dim: int, common_dim: int):
        super().__init__()
        self.projection = ProjectionLayer(teacher_dim, common_dim)
    
    def forward(self, hidden_states):
        """Project teacher hidden states."""
        return self.projection(hidden_states)


class StudentProjection(nn.Module):
    """Student projection layer (trainable)."""
    
    def __init__(self, student_dim: int, common_dim: int):
        super().__init__()
        self.projection = ProjectionLayer(student_dim, common_dim)
    
    def forward(self, hidden_states):
        """Project student hidden states."""
        return self.projection(hidden_states)
