#!/usr/bin/env python3
"""
Extract first 100k samples from medicalGuideline dataset.
"""

import json
from pathlib import Path

def extract_first_n_samples(source_file: str, output_file: str, n: int = 100000):
    """Extract first n samples from a dataset."""
    
    print(f"Loading dataset: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  ✓ Loaded {len(data)} total samples")
    
    # Extract first n
    extracted = data[:n]
    print(f"\nExtracting first {n} samples...")
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved {len(extracted)} samples to {output_file}")
    print()
    print(f"New file: {output_file}")
    print(f"Size: {len(extracted)} samples")

if __name__ == "__main__":
    extract_first_n_samples(
        source_file="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa.json",
        output_file="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_100k.json",
        n=100000,
    )
