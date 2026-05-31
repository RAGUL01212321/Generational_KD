#!/usr/bin/env python3
"""
Create validation set from independent dataset source.
Uses medicalPaper instead of medicalGuideline to ensure zero data leakage.
"""

import json
import hashlib
import random
from pathlib import Path

def compute_sample_hash(question: str, answer: str) -> str:
    """Compute SHA256 hash of a Q&A pair."""
    combined = f"{question.strip()}|{answer.strip()}"
    return hashlib.sha256(combined.encode()).hexdigest()

def create_independent_validation_set(
    train_file: str,
    val_source_file: str,
    output_val: str,
    num_val_samples: int = 2000,
    seed: int = 42,
):
    """
    Create validation set from independent dataset source.
    
    Args:
        train_file: Training dataset file
        val_source_file: Source file for validation samples (different domain)
        output_val: Output validation file path
        num_val_samples: Number of validation samples to extract
        seed: Random seed
    """
    random.seed(seed)
    
    print("="*80)
    print("CREATING INDEPENDENT VALIDATION SET")
    print("="*80)
    print()
    
    # Load training hashes to filter out any overlaps
    print(f"Loading training data: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_hashes = set()
    for q, a in train_data:
        h = compute_sample_hash(q, a)
        train_hashes.add(h)
    
    print(f"  ✓ Loaded {len(train_data)} training samples")
    print(f"  ✓ Computed {len(train_hashes)} training hashes")
    
    # Load validation source
    print(f"\nLoading validation source: {val_source_file}")
    with open(val_source_file, 'r', encoding='utf-8') as f:
        val_source = json.load(f)
    
    print(f"  ✓ Loaded {len(val_source)} source samples")
    
    # Filter out any samples that overlap with training
    print(f"\nFiltering out training set overlaps...")
    val_candidates = []
    overlaps_found = 0
    
    for q, a in val_source:
        h = compute_sample_hash(q, a)
        if h not in train_hashes:
            val_candidates.append((q, a))
        else:
            overlaps_found += 1
    
    if overlaps_found > 0:
        print(f"  ⚠️  Filtered out {overlaps_found} overlapping samples")
    else:
        print(f"  ✓ No overlaps found with training set")
    
    print(f"  ✓ Available unique samples: {len(val_candidates)}")
    
    # Check if we have enough samples
    if len(val_candidates) < num_val_samples:
        print(f"\n⚠️  WARNING: Only {len(val_candidates)} unique samples available,")
        print(f"   but requested {num_val_samples}.")
        print(f"   Using all {len(val_candidates)} available samples.")
        num_val_samples = len(val_candidates)
    
    # Sample validation set
    print(f"\nSampling {num_val_samples} validation samples...")
    val_data = random.sample(val_candidates, num_val_samples)
    
    # Save validation set
    Path(output_val).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_val, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved {len(val_data)} validation samples to {output_val}")
    
    # Verify no leakage
    print(f"\nVerifying no data leakage...")
    val_hashes = set()
    for q, a in val_data:
        h = compute_sample_hash(q, a)
        val_hashes.add(h)
    
    overlaps = train_hashes & val_hashes
    if overlaps:
        print(f"  ❌ ERROR: Found {len(overlaps)} overlapping samples!")
        return False
    else:
        print(f"  ✓ PASS: Zero overlaps with training set")
    
    print()
    print("="*80)
    print(f"✓ Validation set created: {output_val}")
    print(f"  Source domain: Medical Papers (independent from Guidelines)")
    print(f"  Samples: {len(val_data)}")
    print(f"  Data leakage: None ✓")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = create_independent_validation_set(
        train_file="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json",
        val_source_file="Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json",
        output_val="Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa_2000_val.json",
        num_val_samples=2000,
        seed=42,
    )
    
    exit(0 if success else 1)
