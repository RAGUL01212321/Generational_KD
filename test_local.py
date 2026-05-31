#!/usr/bin/env python3
"""
Test script to validate the Gen_KD training pipeline locally.
Tests with small public models before running full training.
"""

import os
import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment():
    """Test basic environment and dependencies."""
    logger.info("\n" + "="*70)
    logger.info("TESTING ENVIRONMENT")
    logger.info("="*70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"✓ PyTorch device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Check imports
    try:
        from Gen_KD.models import ModelWrapper, load_tokenizer
        from Gen_KD.config import GenKDConfig
        from Gen_KD.trainer import GenKDTrainer
        logger.info("✓ Gen_KD modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Gen_KD modules: {e}")
        return False
    
    return True


def test_model_loading(model_name: str = "gpt2"):
    """Test loading a model from HuggingFace."""
    logger.info("\n" + "="*70)
    logger.info(f"TESTING MODEL LOADING: {model_name}")
    logger.info("="*70)
    
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        logger.info(f"Loading config for {model_name}...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"✓ Config loaded")
        logger.info(f"  Model type: {config.model_type}")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Tokenizer loaded")
        logger.info(f"  Vocab size: {len(tokenizer)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False


def test_training_setup():
    """Test training configuration and data loading."""
    logger.info("\n" + "="*70)
    logger.info("TESTING TRAINING SETUP")
    logger.info("="*70)
    
    try:
        from Gen_KD.config import GenKDConfig
        from Gen_KD.models import ModelWrapper
        from transformers import AutoTokenizer
        from datasets import load_dataset
        
        # Create config with small models
        logger.info("Creating training config...")
        config = GenKDConfig(
            model_names=["gpt2", "gpt2", "gpt2"],
            common_dim=128,
            batch_size=2,
            max_seq_len=64,
            epochs=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"✓ Config created for {config.num_generations} generations")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✓ Tokenizer loaded with vocab size {len(tokenizer)}")
        
        # Try loading a small dataset
        logger.info("Loading dataset...")
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            ds = ds.filter(lambda x: len(x['text'].strip()) > 0)
            ds = ds.select(range(min(10, len(ds))))  # Use only 10 samples
        except Exception as e:
            logger.warning(f"  Could not load wikitext dataset, using synthetic data: {e}")
            # Use synthetic data instead
            import datasets
            ds = datasets.Dataset.from_dict({
                'text': [
                    "This is a test document for language model training.",
                    "Natural language processing is important for machine learning.",
                    "Transformers have revolutionized deep learning.",
                    "Knowledge distillation helps compress large models.",
                    "Generational distillation is a novel training approach.",
                ] * 2  # Repeat to get 10 samples
            })
        logger.info(f"✓ Dataset loaded with {len(ds)} samples")
        
        # Tokenize
        def tokenize_fn(examples):
            tokens = tokenizer(
                examples['text'],
                max_length=64,
                padding="max_length",
                truncation=True,
            )
            return tokens
        
        ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format("torch")
        logger.info(f"✓ Dataset tokenized")
        
        # Try creating a dataloader
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=2)
        batch = next(iter(dl))
        batch_shape = batch['input_ids'].shape if hasattr(batch['input_ids'], 'shape') else len(batch['input_ids'])
        logger.info(f"✓ DataLoader working, batch shape: {batch_shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Training setup failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n\n" + "="*70)
    logger.info("GEN_KD LOCAL EXECUTION TEST SUITE")
    logger.info("="*70)
    
    results = {}
    
    # Test 1: Environment
    results['Environment'] = test_environment()
    
    # Test 2: Model Loading
    results['Model Loading'] = test_model_loading("gpt2")
    
    # Test 3: Training Setup
    results['Training Setup'] = test_training_setup()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n" + "="*70)
        logger.info("✓ ALL TESTS PASSED - READY FOR LOCAL TRAINING!")
        logger.info("="*70)
        logger.info("\nTo start training, run:")
        logger.info("  python -m Gen_KD.train --dry-run")
        logger.info("\nOr with custom models:")
        logger.info("  python -m Gen_KD.train \\")
        logger.info("    --models gpt2 gpt2 gpt2 \\")
        logger.info("    --epochs 1 --batch-size 4 --dry-run")
    else:
        logger.error("\n✗ Some tests failed. Please review the output above.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
