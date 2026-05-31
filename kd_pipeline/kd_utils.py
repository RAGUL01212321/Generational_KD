"""
Utilities for KD training: evaluation, inference, etc.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from kd_config import KDConfig
from kd_projection import StudentProjection


class DistilledStudent:
    """Wrapper for distilled student model."""
    
    def __init__(
        self,
        student_model_id: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        
        # Load student model
        self.model = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load projection layer if checkpoint provided
        self.projection = None
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model_id,
            trust_remote_code=True
        )
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load distilled weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load student weights
        self.model.load_state_dict(checkpoint["student_state_dict"])
        
        # Load projection layer
        student_dim = self.model.config.hidden_size
        common_dim = checkpoint["config"]["common_dim"]
        
        self.projection = StudentProjection(student_dim, common_dim)
        self.projection.to(self.device)
        self.projection.load_state_dict(checkpoint["proj_student_state_dict"])
        self.projection.eval()
        
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    @torch.no_grad()
    def extract_hidden_states(self, text: str):
        """Extract hidden states for analysis."""
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )
        
        return outputs.hidden_states[-1]  # Last layer


def compare_models(
    text: str,
    teacher_model_id: str,
    distilled_checkpoint: str,
    device: str = "cuda",
):
    """Compare teacher and distilled student outputs."""
    
    print("\n" + "=" * 70)
    print("COMPARING TEACHER AND DISTILLED STUDENT")
    print("=" * 70)
    
    # Load teacher
    print("\nLoading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    teacher.to(device)
    teacher.eval()
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_id,
        trust_remote_code=True
    )
    
    # Load distilled student
    print("Loading distilled student...")
    student = DistilledStudent(
        "FreedomIntelligence/Apollo-0.5B",
        checkpoint_path=distilled_checkpoint,
        device=device,
    )
    
    # Generate responses
    print("\n" + "-" * 70)
    print(f"Prompt: {text}")
    print("-" * 70)
    
    with torch.no_grad():
        inputs_teacher = teacher_tokenizer(text, return_tensors="pt").to(device)
        teacher_output = teacher.generate(
            **inputs_teacher,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
        teacher_response = teacher_tokenizer.decode(
            teacher_output[0],
            skip_special_tokens=True
        )
        if teacher_response.startswith(text):
            teacher_response = teacher_response[len(text):].strip()
        
        student_response = student.generate(text, max_length=100)
    
    print(f"\nTeacher (1.8B):\n{teacher_response}")
    print(f"\nDistilled Student (0.5B):\n{student_response}")
    print("\n" + "=" * 70 + "\n")


def calculate_parameter_reduction(teacher_id: str, student_id: str) -> float:
    """Calculate parameter reduction percentage."""
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(student_id, trust_remote_code=True)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    reduction = (1 - student_params / teacher_params) * 100
    
    print(f"\nParameter Reduction:")
    print(f"  Teacher (1.8B): {teacher_params:,} params")
    print(f"  Student (0.5B): {student_params:,} params")
    print(f"  Reduction: {reduction:.1f}%\n")
    
    return reduction
