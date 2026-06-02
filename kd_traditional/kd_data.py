"""
Data loading utilities for KD training.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class ApolloKDDataset(Dataset):
    """Dataset for KD training using JSON corpus."""
    
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        seq_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples
        self.samples = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from JSON file or directory of JSON files."""
        print(f"Loading data from {self.dataset_path}...")
        
        if self.dataset_path.is_file():
            # Single file
            self._load_single_file(self.dataset_path)
        elif self.dataset_path.is_dir():
            # Directory of files
            json_files = list(self.dataset_path.glob("*.json"))
            for json_file in json_files:
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                self._load_single_file(json_file)
        else:
            raise FileNotFoundError(f"Path not found: {self.dataset_path}")
        
        if self.max_samples:
            self.samples = self.samples[:self.max_samples]
        
        print(f"Loaded {len(self.samples)} samples total\n")
    
    def _load_single_file(self, json_file: Path):
        """Load data from a single JSON file."""
        print(f"  Reading {json_file.name}...", end="")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.samples.extend(data)
        elif isinstance(data, dict):
            self.samples.append(data)
        
        print(f" ({len(self.samples)} total)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract text based on format (could be different keys)
        if isinstance(sample, dict):
            text = sample.get("text") or sample.get("content") or str(sample)
        else:
            text = str(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def create_dataloader(
    dataset_path: str,
    tokenizer,
    batch_size: int = 4,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
):
    """Create a DataLoader for KD training."""
    dataset = ApolloKDDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_samples=max_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    
    return dataloader, dataset
