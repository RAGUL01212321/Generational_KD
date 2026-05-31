#!/usr/bin/env python3
"""
Comprehensive Validation Script
Evaluates base student vs distilled student on validation set.

Expected Results:
  distilled_student_kd_loss < base_student_kd_loss
  distilled_student_ce_loss <= base_student_ce_loss
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kd_pipeline"))

from validation_config import ValidationConfig
from validation_metrics import validate_student, ValidationMetrics
from qualitative_eval import QualitativeEvaluator, save_qualitative_results
from kd_projection import TeacherProjection, StudentProjection
from kd_data import create_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Main validation pipeline."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create results directory
        Path(config.results_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*70)
        logger.info("KNOWLEDGE DISTILLATION VALIDATION PIPELINE")
        logger.info("="*70)
    
    def load_models(self):
        """Load teacher and student models."""
        logger.info("\nLoading models...")
        
        # Load teacher (frozen)
        logger.info(f"Loading teacher: {self.config.teacher_model_id}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.teacher.eval()
        
        # Load base student (fresh)
        logger.info(f"Loading base student: {self.config.student_model_id}")
        self.base_student = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.base_student.eval()
        
        # Load distilled student
        logger.info(f"Loading distilled student: {self.config.distilled_student_checkpoint}")
        checkpoint = torch.load(
            self.config.distilled_student_checkpoint,
            map_location=self.device
        )
        self.distilled_student = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.distilled_student.load_state_dict(checkpoint['student_state_dict'])
        self.distilled_student.eval()
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("✓ Models loaded successfully")
    
    def load_projections(self):
        """Load or create projection layers."""
        logger.info("\nSetting up projections...")
        
        teacher_dim = self.teacher.config.hidden_size
        student_dim = self.base_student.config.hidden_size
        common_dim = self.config.common_dim
        
        logger.info(f"  Teacher dim: {teacher_dim}")
        logger.info(f"  Student dim: {student_dim}")
        logger.info(f"  Common dim: {common_dim}")
        
        # Create projections
        self.proj_teacher = TeacherProjection(teacher_dim, common_dim).to(self.device)
        self.proj_student = StudentProjection(student_dim, common_dim).to(self.device)
        
        # Load checkpoint with projection states
        checkpoint = torch.load(
            self.config.distilled_student_checkpoint,
            map_location=self.device
        )
        
        # Load distilled student projection
        if 'proj_student_state_dict' in checkpoint:
            self.proj_student.load_state_dict(checkpoint['proj_student_state_dict'])
            logger.info("  ✓ Loaded distilled student projection")
        
        # Load teacher projection (IMPORTANT: same weights used during training)
        if 'proj_teacher_state_dict' in checkpoint:
            self.proj_teacher.load_state_dict(checkpoint['proj_teacher_state_dict'])
            logger.info("  ✓ Loaded training teacher projection (ensures matching KD loss)")
        else:
            logger.warning("  ⚠ Teacher projection not found in checkpoint - KD loss will not match training!")
        
        self.proj_teacher.eval()
        self.proj_student.eval()
    
    def create_validation_dataloader(self):
        """Create validation dataloader."""
        logger.info(f"\nLoading validation data: {self.config.validation_dataset_file}")
        
        self.val_dataloader, self.val_dataset = create_dataloader(
            dataset_path=self.config.validation_dataset_file,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            max_samples=None,
        )
        
        logger.info(f"  ✓ Validation dataset size: {len(self.val_dataset)}")
    
    def run_quantitative_evaluation(self) -> Dict:
        """Run quantitative evaluation."""
        logger.info("\n" + "="*70)
        logger.info("QUANTITATIVE EVALUATION")
        logger.info("="*70)
        
        # Validate base student
        logger.info("\nEvaluating base student...")
        base_metrics = validate_student(
            student=self.base_student,
            teacher=self.teacher,
            dataloader=self.val_dataloader,
            proj_student=self.proj_student,
            proj_teacher=self.proj_teacher,
            device=str(self.device),
        )
        
        logger.info(f"  Base Student KD Loss:     {base_metrics['kd_loss']:.4f}")
        logger.info(f"  Base Student CE Loss:     {base_metrics['ce_loss']:.4f}")
        logger.info(f"  Base Student Perplexity:  {base_metrics['perplexity']:.4f}")
        logger.info(f"  Base Student Cosine Sim:  {base_metrics['cosine_similarity']:.4f}")
        
        # Validate distilled student
        logger.info("\nEvaluating distilled student...")
        distilled_metrics = validate_student(
            student=self.distilled_student,
            teacher=self.teacher,
            dataloader=self.val_dataloader,
            proj_student=self.proj_student,
            proj_teacher=self.proj_teacher,
            device=str(self.device),
        )
        
        logger.info(f"  Distilled Student KD Loss:     {distilled_metrics['kd_loss']:.4f}")
        logger.info(f"  Distilled Student CE Loss:     {distilled_metrics['ce_loss']:.4f}")
        logger.info(f"  Distilled Student Perplexity:  {distilled_metrics['perplexity']:.4f}")
        logger.info(f"  Distilled Student Cosine Sim:  {distilled_metrics['cosine_similarity']:.4f}")
        
        # Compare
        logger.info("\n" + "-"*70)
        logger.info("COMPARISON")
        logger.info("-"*70)
        
        kd_improvement = base_metrics['kd_loss'] - distilled_metrics['kd_loss']
        kd_improvement_pct = (kd_improvement / base_metrics['kd_loss']) * 100
        
        ce_improvement = base_metrics['ce_loss'] - distilled_metrics['ce_loss']
        ce_improvement_pct = (ce_improvement / base_metrics['ce_loss']) * 100
        
        cos_sim_improvement = distilled_metrics['cosine_similarity'] - base_metrics['cosine_similarity']
        cos_sim_improvement_pct = (cos_sim_improvement / base_metrics['cosine_similarity']) * 100 if base_metrics['cosine_similarity'] != 0 else 0
        
        logger.info(f"KD Loss improvement:        {kd_improvement:.4f} ({kd_improvement_pct:+.1f}%)")
        logger.info(f"CE Loss improvement:        {ce_improvement:.4f} ({ce_improvement_pct:+.1f}%)")
        logger.info(f"Cosine Similarity improvement: {cos_sim_improvement:+.4f} ({cos_sim_improvement_pct:+.1f}%)")
        
        # Expected results
        logger.info("\n" + "-"*70)
        logger.info("EXPECTED RESULTS")
        logger.info("-"*70)
        
        kd_passed = distilled_metrics['kd_loss'] < base_metrics['kd_loss']
        ce_passed = distilled_metrics['ce_loss'] <= base_metrics['ce_loss']
        cos_sim_passed = distilled_metrics['cosine_similarity'] > base_metrics['cosine_similarity']
        
        logger.info(f"KD Loss (distilled < base):        {'✓ PASS' if kd_passed else '✗ FAIL'}")
        logger.info(f"CE Loss (distilled ≤ base):        {'✓ PASS' if ce_passed else '✗ FAIL'}")
        logger.info(f"Cosine Sim (distilled > base):     {'✓ PASS' if cos_sim_passed else '✗ FAIL'}")
        
        results = {
            "quantitative": {
                "base_student": {
                    "kd_loss": base_metrics['kd_loss'],
                    "ce_loss": base_metrics['ce_loss'],
                    "perplexity": base_metrics['perplexity'],
                    "cosine_similarity": base_metrics['cosine_similarity'],
                },
                "distilled_student": {
                    "kd_loss": distilled_metrics['kd_loss'],
                    "ce_loss": distilled_metrics['ce_loss'],
                    "perplexity": distilled_metrics['perplexity'],
                    "cosine_similarity": distilled_metrics['cosine_similarity'],
                },
                "improvements": {
                    "kd_loss_diff": kd_improvement,
                    "kd_loss_improvement_pct": kd_improvement_pct,
                    "ce_loss_diff": ce_improvement,
                    "ce_loss_improvement_pct": ce_improvement_pct,
                    "cosine_similarity_diff": cos_sim_improvement,
                    "cosine_similarity_improvement_pct": cos_sim_improvement_pct,
                },
                "tests_passed": {
                    "kd_loss_test": kd_passed,
                    "ce_loss_test": ce_passed,
                    "cosine_similarity_test": cos_sim_passed,
                },
            }
        }
        
        return results
    
    def run_qualitative_evaluation(self) -> Dict:
        """Run qualitative evaluation."""
        logger.info("\n" + "="*70)
        logger.info("QUALITATIVE EVALUATION")
        logger.info("="*70)
        
        evaluator = QualitativeEvaluator(device=str(self.device))
        
        # Sample validation data (questions only)
        samples = [
            (item[0], item[1]) for item in self.val_dataset.samples[:self.config.num_qualitative_samples]
        ]
        
        qual_results = evaluator.evaluate_generation_quality(
            base_student=self.base_student,
            distilled_student=self.distilled_student,
            tokenizer=self.tokenizer,
            validation_samples=samples,
            num_samples=len(samples),
        )
        
        return {"qualitative": qual_results}
    
    def run_full_validation(self, run_qualitative: bool = True) -> Dict:
        """Run full validation pipeline."""
        try:
            self.load_models()
            self.load_projections()
            self.create_validation_dataloader()
            
            # Quantitative evaluation
            results = self.run_quantitative_evaluation()
            
            # Qualitative evaluation
            if run_qualitative:
                logger.info("\nSkipping qualitative evaluation (set --with-qualitative to enable)")
            else:
                qual_results = self.run_qualitative_evaluation()
                results.update(qual_results)
            
            # Save results
            self.save_results(results)
            
            return results
        
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            raise
    
    def save_results(self, results: Dict):
        """Save validation results to file."""
        output_file = self.config.results_file
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Knowledge Distillation Pipeline"
    )
    
    parser.add_argument(
        "--validation-file",
        type=str,
        default="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_validation.json",
        help="Validation dataset file"
    )
    parser.add_argument(
        "--distilled-checkpoint",
        type=str,
        default="kd_checkpoints/final.pt",
        help="Path to distilled student checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Validation batch size"
    )
    parser.add_argument(
        "--with-qualitative",
        action="store_true",
        help="Run qualitative evaluation (slow)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ValidationConfig(
        validation_dataset_file=args.validation_file,
        distilled_student_checkpoint=args.distilled_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Run validation
    pipeline = ValidationPipeline(config)
    results = pipeline.run_full_validation(run_qualitative=not args.with_qualitative)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
