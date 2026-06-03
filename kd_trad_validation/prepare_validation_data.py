#!/usr/bin/env python3
"""
Prepare validation dataset from full dataset.
Creates held-out validation set that was NOT used in training.
"""

import json
import random
import argparse
from pathlib import Path


def create_validation_set(
    source_file: str,
    training_file: str,
    output_file: str,
    validation_size: int = 1000,
    seed: int = 42,
):
    """
    Create validation set from source, excluding training samples.
    
    Args:
        source_file: Full dataset file
        training_file: Training dataset file
        output_file: Output validation file
        validation_size: Number of validation samples
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"Loading full dataset from {source_file}...")
    with open(source_file, 'r') as f:
        full_data = json.load(f)
    
    print(f"Loading training dataset from {training_file}...")
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    # Convert training data to set of tuples for fast lookup
    training_set = set()
    for item in training_data:
        if isinstance(item, list) and len(item) >= 2:
            training_set.add((item[0], item[1]))
    
    print(f"Training set has {len(training_set)} unique samples")
    
    # Filter out training samples
    validation_candidates = []
    for item in full_data:
        if isinstance(item, list) and len(item) >= 2:
            if (item[0], item[1]) not in training_set:
                validation_candidates.append(item)
    
    print(f"Found {len(validation_candidates)} non-training samples")
    
    # Sample validation set
    if len(validation_candidates) < validation_size:
        print(f"Warning: Only {len(validation_candidates)} non-training samples available")
        print(f"  Requested {validation_size}. Using all available.")
        validation_data = validation_candidates
    else:
        validation_data = random.sample(validation_candidates, validation_size)
    
    # Save validation set
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"✓ Created validation set with {len(validation_data)} samples")
    print(f"  Saved to: {output_file}")
    
    return validation_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare validation dataset for traditional KD"
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa.json",
        help="Full dataset file"
    )
    parser.add_argument(
        "--training-file",
        type=str,
        default="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json",
        help="Training dataset file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10k_val.json",
        help="Output validation file"
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Check if files exist or try relative paths from parent directory if needed
    source_path = Path(args.source_file)
    if not source_path.exists():
        parent_path = Path("..") / args.source_file
        if parent_path.exists():
            args.source_file = str(parent_path)
            
    training_path = Path(args.training_file)
    if not training_path.exists():
        parent_path = Path("..") / args.training_file
        if parent_path.exists():
            args.training_file = str(parent_path)
            
    output_path = Path(args.output_file)
    if not output_path.parent.exists() and (Path("..") / args.output_file).parent.exists():
        args.output_file = str(Path("..") / args.output_file)
    
    create_validation_set(
        source_file=args.source_file,
        training_file=args.training_file,
        output_file=args.output_file,
        validation_size=args.validation_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
