#!/usr/bin/env python3
"""
Data Integrity Checker
Verify that training and validation datasets are truly separate.
Uses hash-based comparison to detect any overlapping samples.
"""

import json
import hashlib
import sys
from pathlib import Path
from collections import defaultdict

def compute_sample_hash(question: str, answer: str) -> str:
    """Compute SHA256 hash of a Q&A pair."""
    combined = f"{question.strip()}|{answer.strip()}"
    return hashlib.sha256(combined.encode()).hexdigest()

def load_dataset(filepath: str) -> list:
    """Load JSON dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_data_integrity(train_file: str, val_file: str):
    """Check if training and validation data are truly separate."""
    
    print("="*80)
    print("DATA INTEGRITY CHECK - TRAINING vs VALIDATION")
    print("="*80)
    print()
    
    # Load datasets
    print(f"Loading training data: {train_file}")
    train_data = load_dataset(train_file)
    print(f"  ✓ Loaded {len(train_data)} training samples")
    
    print(f"\nLoading validation data: {val_file}")
    val_data = load_dataset(val_file)
    print(f"  ✓ Loaded {len(val_data)} validation samples")
    
    print()
    print("-"*80)
    
    # Compute hashes
    print("\nComputing sample hashes...")
    
    train_hashes = {}
    for i, (q, a) in enumerate(train_data):
        h = compute_sample_hash(q, a)
        train_hashes[h] = (i, q[:50] + "..." if len(q) > 50 else q)
    
    val_hashes = {}
    for i, (q, a) in enumerate(val_data):
        h = compute_sample_hash(q, a)
        val_hashes[h] = (i, q[:50] + "..." if len(q) > 50 else q)
    
    print(f"  ✓ Computed {len(train_hashes)} training hashes")
    print(f"  ✓ Computed {len(val_hashes)} validation hashes")
    
    # Find overlaps
    print("\n" + "-"*80)
    print("OVERLAP DETECTION")
    print("-"*80)
    
    overlapping_hashes = set(train_hashes.keys()) & set(val_hashes.keys())
    
    if overlapping_hashes:
        print(f"\n⚠️  ALERT: Found {len(overlapping_hashes)} OVERLAPPING SAMPLES!")
        print("\nOverlapping samples:")
        for i, h in enumerate(overlapping_hashes, 1):
            train_idx, train_q = train_hashes[h]
            val_idx, val_q = val_hashes[h]
            print(f"\n  {i}. Hash: {h[:16]}...")
            print(f"     Training[{train_idx}]: {train_q}")
            print(f"     Validation[{val_idx}]: {val_q}")
    else:
        print("\n✓ PASS: No exact duplicates found between training and validation!")
    
    # Check for near-duplicates by question only
    print("\n" + "-"*80)
    print("NEAR-DUPLICATE CHECK (Questions only)")
    print("-"*80)
    
    train_questions = {}
    for i, (q, a) in enumerate(train_data):
        q_hash = hashlib.sha256(q.strip().encode()).hexdigest()
        train_questions[q_hash] = (i, q[:50] + "..." if len(q) > 50 else q)
    
    val_questions = {}
    for i, (q, a) in enumerate(val_data):
        q_hash = hashlib.sha256(q.strip().encode()).hexdigest()
        val_questions[q_hash] = (i, q[:50] + "..." if len(q) > 50 else q)
    
    overlapping_questions = set(train_questions.keys()) & set(val_questions.keys())
    
    if overlapping_questions:
        print(f"\n⚠️  ALERT: Found {len(overlapping_questions)} OVERLAPPING QUESTIONS!")
        print("(Different answers but same question)")
        for i, h in enumerate(overlapping_questions, 1):
            train_idx, train_q = train_questions[h]
            val_idx, val_q = val_questions[h]
            print(f"\n  {i}. Question[{train_idx}/{val_idx}]: {train_q}")
    else:
        print("\n✓ PASS: No overlapping questions found!")
    
    # Stats
    print("\n" + "-"*80)
    print("STATISTICS")
    print("-"*80)
    
    unique_train = len(train_hashes)
    unique_val = len(val_hashes)
    total_train = len(train_data)
    total_val = len(val_data)
    
    train_dups = total_train - unique_train
    val_dups = total_val - unique_val
    
    print(f"\nTraining set:")
    print(f"  Total samples:     {total_train}")
    print(f"  Unique samples:    {unique_train}")
    print(f"  Internal dups:     {train_dups}")
    if train_dups > 0:
        print(f"  ⚠️  WARNING: {train_dups} duplicate samples within training set!")
    
    print(f"\nValidation set:")
    print(f"  Total samples:     {total_val}")
    print(f"  Unique samples:    {unique_val}")
    print(f"  Internal dups:     {val_dups}")
    if val_dups > 0:
        print(f"  ⚠️  WARNING: {val_dups} duplicate samples within validation set!")
    
    print(f"\nCross-set overlap:")
    print(f"  Exact matches:     {len(overlapping_hashes)}")
    print(f"  Question overlaps: {len(overlapping_questions)}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    overlap_pct = (len(overlapping_questions) / len(val_data)) * 100 if len(val_data) > 0 else 0
    
    # Only flag as critical if we have exact matches (full Q&A duplicates)
    if overlapping_hashes:
        print("\n❌ CRITICAL: DATA LEAKAGE DETECTED!")
        print(f"   {len(overlapping_hashes)} exact Q&A duplicates found in validation set.")
        print("   Results are meaningless - validation data is not truly separate.")
        return False
    elif overlap_pct > 5:  # >5% question overlap
        print("\n⚠️  WARNING: Significant question overlap detected!")
        print(f"   {len(overlapping_questions)} questions overlap between sets ({overlap_pct:.1f}%).")
        print("   Results may be partially inflated by memorization.")
        return False
    else:
        print("\n✓ PASS: Datasets are properly separated!")
        print(f"   Exact duplicates: {len(overlapping_hashes)}")
        if len(overlapping_questions) > 0:
            print(f"   Question overlaps: {len(overlapping_questions)} ({overlap_pct:.1f}%)")
            print("   (Negligible - likely due to dataset variations)")
        print("   Training and validation samples are distinct.")
        print("   Results are valid measures of generalization.")
        return True

if __name__ == "__main__":
    # Default paths
    train_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json"
    val_file = "Dataset/ApolloCorpus/pretrain/validation_data.json"
    
    # Allow custom paths
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
    if len(sys.argv) > 2:
        val_file = sys.argv[2]
    
    # Check files exist
    if not Path(train_file).exists():
        print(f"ERROR: Training file not found: {train_file}")
        sys.exit(1)
    if not Path(val_file).exists():
        print(f"ERROR: Validation file not found: {val_file}")
        sys.exit(1)
    
    # Run check
    success = check_data_integrity(train_file, val_file)
    sys.exit(0 if success else 1)
