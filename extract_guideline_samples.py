#!/usr/bin/env python3
"""
Extract training and validation samples from full medical guideline dataset.
Since total is ~99.6k: uses ~89.6k for training, ~10k for validation.
Ensures no overlap between training and validation sets.
"""

import json
from pathlib import Path

def extract_samples():
    """Extract training and validation samples from medicalGuideline dataset."""
    
    source_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa.json"
    train_output = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json"
    val_output = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10k_val.json"
    
    print("="*80)
    print("EXTRACTING SAMPLES FROM FULL MEDICAL GUIDELINE DATASET")
    print("="*80)
    print()
    
    # Load full dataset
    print(f"Loading full dataset: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    total_samples = len(full_data)
    print(f"  ✓ Loaded {total_samples} total samples")
    
    # Calculate split (90/10)
    split_idx = int(total_samples * 0.9)
    
    # Extract training: samples 0 to 90%
    print(f"\nExtracting training samples (0-{split_idx})...")
    train_data = full_data[:split_idx]
    print(f"  ✓ Extracted {len(train_data)} training samples")
    
    # Extract validation: samples 90% to end
    print(f"\nExtracting validation samples ({split_idx}-{total_samples})...")
    val_data = full_data[split_idx:]
    print(f"  ✓ Extracted {len(val_data)} validation samples")
    
    # Save training set
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved training set: {train_output}")
    
    # Save validation set
    Path(val_output).parent.mkdir(parents=True, exist_ok=True)
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved validation set: {val_output}")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training set: {len(train_data)} samples (90% of dataset)")
    print(f"Validation set: {len(val_data)} samples (10% of dataset)")
    print(f"Total: {len(train_data) + len(val_data)} samples")
    print(f"No overlap: ✓")
    print("="*80)

if __name__ == "__main__":
    extract_samples()
