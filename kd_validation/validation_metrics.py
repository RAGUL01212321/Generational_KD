#!/usr/bin/env python3
"""
Validation Metrics - Compute KD loss and CE loss on validation set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationMetrics:
    """Compute validation metrics: KD loss and CE loss."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.mse_loss = nn.MSELoss()
    
    def compute_kd_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        proj_student: nn.Module,
        proj_teacher: nn.Module,
        attention_mask: torch.Tensor,
    ) -> float:
        """
        Compute KD loss between student and teacher hidden states.
        
        Args:
            student_hidden: Student hidden states (batch, seq_len, hidden_size)
            teacher_hidden: Teacher hidden states (batch, seq_len, hidden_size)
            proj_student: Student projection layer
            proj_teacher: Teacher projection layer
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            KD loss value
        """
        # Project to common space
        Z_S = proj_student(student_hidden)  # (batch, seq_len, common_dim)
        Z_T = proj_teacher(teacher_hidden)  # (batch, seq_len, common_dim)
        
        # Pool
        p_S = self._pool_hidden_states(Z_S, attention_mask)  # (batch, common_dim)
        p_T = self._pool_hidden_states(Z_T, attention_mask)  # (batch, common_dim)
        
        # MSE loss
        loss = self.mse_loss(p_S, p_T)
        return loss.item()
    
    def compute_cosine_similarity(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        proj_student: nn.Module,
        proj_teacher: nn.Module,
        attention_mask: torch.Tensor,
    ) -> float:
        """
        Compute cosine similarity between projected student and teacher hidden states.
        
        Args:
            student_hidden: Student hidden states (batch, seq_len, hidden_size)
            teacher_hidden: Teacher hidden states (batch, seq_len, hidden_size)
            proj_student: Student projection layer
            proj_teacher: Teacher projection layer
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            Mean cosine similarity (higher is better, max=1.0)
        """
        # Project to common space
        Z_S = proj_student(student_hidden)  # (batch, seq_len, common_dim)
        Z_T = proj_teacher(teacher_hidden)  # (batch, seq_len, common_dim)
        
        # Pool
        p_S = self._pool_hidden_states(Z_S, attention_mask)  # (batch, common_dim)
        p_T = self._pool_hidden_states(Z_T, attention_mask)  # (batch, common_dim)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(p_S, p_T, dim=-1).mean()
        return cos_sim.item()
    
    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute cross-entropy loss and perplexity.
        
        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            labels: Input IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            (ce_loss, perplexity)
        """
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = shift_logits.shape
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_mask = shift_mask.view(-1)
        
        # CE loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(shift_logits, shift_labels)
        
        # Masked loss
        masked_losses = losses * shift_mask
        ce_loss = masked_losses.sum() / shift_mask.sum()
        
        # Perplexity
        perplexity = torch.exp(ce_loss).item()
        
        return ce_loss.item(), perplexity
    
    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling with attention mask."""
        # Expand mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        pooled = sum_hidden / sum_mask
        return pooled


def validate_checkpoint(
    checkpoint_path: str,
    dataloader,
    teacher_model,
    proj_teacher,
    proj_student,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Validate a distilled student checkpoint.
    
    Args:
        checkpoint_path: Path to distilled student checkpoint
        dataloader: Validation dataloader
        teacher_model: Teacher model (frozen)
        proj_teacher: Teacher projection layer
        proj_student: Student projection layer
        device: Device to use
    
    Returns:
        Dictionary with metrics
    """
    import torch
    from pathlib import Path
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_state = checkpoint['student_state_dict']
    
    # Create student model
    from transformers import AutoModelForCausalLM
    config = checkpoint['config']
    student = AutoModelForCausalLM.from_pretrained(
        config['student_model_id'],
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    student.load_state_dict(student_state)
    student.eval()
    
    # Compute metrics
    metrics = validate_student(
        student=student,
        teacher=teacher_model,
        dataloader=dataloader,
        proj_student=proj_student,
        proj_teacher=proj_teacher,
        device=device,
    )
    
    return metrics


def validate_student(
    student,
    teacher,
    dataloader,
    proj_student,
    proj_teacher,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute validation metrics for student model.
    
    Args:
        student: Student model
        teacher: Teacher model (frozen)
        dataloader: Validation dataloader
        proj_student: Student projection layer
        proj_teacher: Teacher projection layer
        device: Device
    
    Returns:
        Dictionary with metrics
    """
    student.eval()
    teacher.eval()
    proj_student.eval()
    proj_teacher.eval()
    
    metrics = ValidationMetrics(device=device)
    
    total_kd_loss = 0.0
    total_ce_loss = 0.0
    total_perplexity = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Teacher forward
            teacher_output = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_hidden = teacher_output.hidden_states[-1].float()
            
            # Student forward
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            student_hidden = student_output.hidden_states[-1].float()
            student_logits = student_output.logits
            
            # Compute losses
            kd_loss = metrics.compute_kd_loss(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                proj_student=proj_student,
                proj_teacher=proj_teacher,
                attention_mask=attention_mask,
            )
            
            # Compute cosine similarity
            cos_sim = metrics.compute_cosine_similarity(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                proj_student=proj_student,
                proj_teacher=proj_teacher,
                attention_mask=attention_mask,
            )
            
            ce_loss, perplexity = metrics.compute_ce_loss(
                logits=student_logits,
                labels=input_ids,
                attention_mask=attention_mask,
            )
            
            total_kd_loss += kd_loss
            total_ce_loss += ce_loss
            total_perplexity += perplexity
            total_cosine_sim += cos_sim
            num_batches += 1
    
    return {
        "kd_loss": total_kd_loss / num_batches,
        "ce_loss": total_ce_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
        "cosine_similarity": total_cosine_sim / num_batches,
    }
