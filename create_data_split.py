#!/usr/bin/env python3
"""
Create proper train/validation split.
Ensures validation set is truly separate from training set.
"""

import json
import random
from pathlib import Path

def create_train_val_split(
    source_file: str,
    output_train: str,
    output_val: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Split a dataset into train and validation sets.
    
    Args:
        source_file: Original dataset file
        output_train: Output training file path
        output_val: Output validation file path
        train_ratio: Fraction for training (default 0.8 = 80/20 split)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load data
    print(f"Loading dataset: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Shuffle and split
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(output_val).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_train, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(train_data)} training samples to {output_train}")
    
    with open(output_val, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(val_data)} validation samples to {output_val}")
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} validation")
    print(f"Ratio: {(len(train_data)/len(data))*100:.1f}% / {(len(val_data)/len(data))*100:.1f}%")

if __name__ == "__main__":
    # Example: Create 80/20 split from medicalGuideline dataset
    # You can also use other source datasets
    
    create_train_val_split(
        source_file="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json",
        output_train="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_8000.json",
        output_val="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_2000_val.json",
        train_ratio=0.8,
        seed=42,
    )
    
    print("\n✓ Done! Update your config to use the new validation file.")
    print("  Set dataset_file to: medicalGuideline_en_qa_8000.json")
    print("  Set validation file to: medicalGuideline_en_qa_2000_val.json")
