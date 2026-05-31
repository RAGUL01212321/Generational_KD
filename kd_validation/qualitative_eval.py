#!/usr/bin/env python3
"""
Qualitative Evaluation - Compare base and distilled student generations.
"""

import json
import torch
from typing import Dict, List
import random


class QualitativeEvaluator:
    """Generate and compare outputs from base and distilled students."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
    
    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """Generate response from model."""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def evaluate_generation_quality(
        self,
        base_student,
        distilled_student,
        tokenizer,
        validation_samples: List,
        num_samples: int = 20,
    ) -> Dict:
        """
        Compare generations from base and distilled students.
        
        Args:
            base_student: Base student model
            distilled_student: Distilled student model
            tokenizer: Tokenizer
            validation_samples: List of (question, answer) pairs
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary with qualitative results
        """
        base_student.eval()
        distilled_student.eval()
        
        # Sample prompts
        samples = random.sample(
            validation_samples,
            min(num_samples, len(validation_samples))
        )
        
        results = []
        
        with torch.no_grad():
            for idx, (question, ground_truth) in enumerate(samples, 1):
                # Generate from both models
                base_response = self.generate_response(
                    model=base_student,
                    tokenizer=tokenizer,
                    prompt=question,
                )
                
                distilled_response = self.generate_response(
                    model=distilled_student,
                    tokenizer=tokenizer,
                    prompt=question,
                )
                
                result = {
                    "sample_id": idx,
                    "question": question,
                    "ground_truth": ground_truth[:200],  # Truncate for display
                    "base_student": base_response,
                    "distilled_student": distilled_response,
                }
                
                results.append(result)
                
                # Print comparison
                print(f"\n{'='*70}")
                print(f"Sample {idx}/{num_samples}")
                print(f"{'='*70}")
                print(f"Question: {question[:100]}...")
                print(f"\nBase Student:\n{base_response[:300]}...")
                print(f"\nDistilled Student:\n{distilled_response[:300]}...")
                print()
        
        return {
            "num_samples": len(results),
            "samples": results,
        }


def save_qualitative_results(results: Dict, output_file: str):
    """Save qualitative evaluation results."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Qualitative results saved to: {output_file}")


def print_qualitative_summary(results: Dict):
    """Print summary of qualitative evaluation."""
    print("\n" + "="*70)
    print("QUALITATIVE EVALUATION SUMMARY")
    print("="*70)
    print(f"Evaluated {results['num_samples']} samples")
    print("\nSamples saved - review for:")
    print("  - Response relevance and accuracy")
    print("  - Medical terminology alignment")
    print("  - Answer structure and clarity")
    print("  - Comparison: distilled vs base student")
    print("="*70 + "\n")
