#!/usr/bin/env python3
"""
Quick Validation Runner for Traditional KD - Simplest way to validate your distilled model.
"""

import subprocess
import sys
from pathlib import Path


def run_validation_pipeline():
    """Run the complete validation pipeline."""
    
    print("\n" + "="*70)
    print("QUICK VALIDATION RUNNER FOR TRADITIONAL KD")
    print("="*70)
    
    # Check if validation data exists
    val_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10k_val.json"
    
    # Path resolution relative to parent folder if running inside validation folder
    root_dir = Path(__file__).parent.parent
    actual_val_file = root_dir / val_file
    
    if not actual_val_file.exists():
        print(f"\n⚠ Validation dataset not found: {actual_val_file}")
        print("Creating validation dataset...")
        
        result = subprocess.run(
            [
                sys.executable,
                "kd_trad_validation/prepare_validation_data.py",
                "--source-file", "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa.json",
                "--training-file", "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json",
                "--output-file", str(val_file),
                "--validation-size", "1000",
            ],
            cwd=str(root_dir),
        )
        
        if result.returncode != 0:
            print("✗ Failed to create validation dataset")
            return 1
    
    print(f"\n✓ Validation dataset exists: {actual_val_file}")
    
    # Run validation
    print("\nRunning validation...")
    result = subprocess.run(
        [
            sys.executable,
            "kd_trad_validation/run_validation.py",
            "--validation-file", str(val_file),
            "--distilled-checkpoint", "kd_checkpoints/final.pt",
            "--device", "cuda",
            "--batch-size", "4",
        ],
        cwd=str(root_dir),
    )
    
    if result.returncode != 0:
        print("\n✗ Validation failed")
        return 1
    
    print("\n✓ Validation completed successfully!")
    print(f"Results saved to: kd_trad_validation/results/validation_results.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_validation_pipeline())
