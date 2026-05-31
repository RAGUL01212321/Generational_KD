#!/usr/bin/env python3
"""
Download and cache models locally for offline training.
Useful for preloading models before traveling or for backup.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Popular small/medium models for testing
RECOMMENDED_MODELS = {
    'tiny': [
        'sshleifer/tiny-gpt2',
        'distilgpt2',
    ],
    'small': [
        'gpt2',
        'bert-base-uncased',
        'distilbert-base-uncased',
    ],
    'medium': [
        'gpt2-medium',
        'gpt2-large',
        'bert-large-uncased',
    ],
}


def download_model(model_name: str, cache_dir: str = None):
    """Download a model and its tokenizer."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Downloading: {model_name}")
    logger.info(f"{'='*70}")
    
    try:
        # Set cache directory
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
        
        # Download config
        logger.info("Downloading config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info(f"✓ Config downloaded")
        logger.info(f"  Model type: {config.model_type}")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info(f"✓ Tokenizer downloaded ({len(tokenizer)} tokens)")
        
        # Download model (this is the largest download)
        logger.info("Downloading model weights...")
        logger.info("(This may take a few minutes depending on model size)")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info(f"✓ Model downloaded")
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {param_count:,}")
        
        logger.info(f"\n✓ {model_name} successfully cached!")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def download_models_by_category(category: str, cache_dir: str = None):
    """Download all models in a category."""
    if category not in RECOMMENDED_MODELS:
        logger.error(f"Unknown category: {category}")
        logger.error(f"Available categories: {', '.join(RECOMMENDED_MODELS.keys())}")
        return False
    
    models = RECOMMENDED_MODELS[category]
    logger.info(f"\nDownloading {len(models)} {category} models...")
    
    results = {}
    for model_name in models:
        results[model_name] = download_model(model_name, cache_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download models for offline training"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--models',
        nargs='+',
        help='Specific model names to download'
    )
    group.add_argument(
        '--category',
        choices=['tiny', 'small', 'medium'],
        help='Download all models in a category'
    )
    
    parser.add_argument(
        '--cache-dir',
        default=None,
        help='Directory to cache models (default: $HF_HOME)'
    )
    
    args = parser.parse_args()
    
    # Determine cache directory
    cache_dir = args.cache_dir or os.environ.get('HF_HOME')
    if cache_dir:
        logger.info(f"Using cache directory: {cache_dir}")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Download models
    if args.category:
        results = download_models_by_category(args.category, cache_dir)
    else:
        results = {}
        for model_name in args.models:
            results[model_name] = download_model(model_name, cache_dir)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*70}")
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    logger.info(f"Successfully downloaded: {success_count}/{total_count}")
    
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {model_name}")
    
    if cache_dir:
        logger.info(f"\nModels cached in: {cache_dir}")
    
    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
