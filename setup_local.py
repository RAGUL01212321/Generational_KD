#!/usr/bin/env python3
"""
Setup script to prepare the Gen_KD project for local model execution.
Downloads models, validates installations, and sets up the environment.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalSetup:
    """Handles setup and validation for local model execution."""
    
    def __init__(self, root_dir: str = None):
        """Initialize setup with project root directory."""
        if root_dir is None:
            root_dir = Path(__file__).parent
        self.root_dir = Path(root_dir)
        self.models_dir = self.root_dir / "models"
        self.config_dir = self.root_dir / "model_configs"
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Project root: {self.root_dir}")
        logger.info(f"Models cache dir: {self.models_dir}")
    
    def check_dependencies(self) -> bool:
        """Verify all required packages are installed."""
        logger.info("\n" + "="*70)
        logger.info("CHECKING DEPENDENCIES")
        logger.info("="*70)
        
        required_packages = {
            'torch': torch.__version__,
            'transformers': None,
            'datasets': None,
            'accelerate': None,
            'numpy': None,
        }
        
        all_installed = True
        for package in required_packages:
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                logger.info(f"✓ {package:<20} {version}")
            except ImportError:
                logger.error(f"✗ {package:<20} NOT INSTALLED")
                all_installed = False
        
        if not all_installed:
            logger.error("\nSome packages are missing. Install with:")
            logger.error(f"  pip install -r {self.root_dir / 'Gen_KD' / 'requirements.txt'}")
            return False
        
        # Check CUDA availability
        logger.info(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.get_device_name(0)}")
        
        return True
    
    def read_model_config(self, config_path: Path) -> Dict:
        """Read a model configuration file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_model_identifier(self, config: Dict) -> str:
        """Extract the model identifier/path from config."""
        name_or_path = config.get('_name_or_path', '')
        model_type = config.get('model_type', 'unknown')
        return name_or_path or model_type
    
    def download_model(self, model_identifier: str, cache_dir: str = None) -> Tuple[bool, str]:
        """
        Attempt to download a model from HuggingFace or local path.
        
        Returns:
            (success: bool, message: str)
        """
        logger.info(f"\nAttempting to load model: {model_identifier}")
        
        try:
            # Try loading from HuggingFace Hub
            config = AutoConfig.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            logger.info(f"✓ Config loaded successfully")
            logger.info(f"  Model type: {config.model_type}")
            logger.info(f"  Hidden size: {config.hidden_size}")
            logger.info(f"  Vocab size: {config.vocab_size}")
            
            # Download tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_identifier,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                logger.info(f"✓ Tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"⚠ Could not load tokenizer: {e}")
            
            return True, f"Model {model_identifier} is accessible"
        
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"⚠ Could not load {model_identifier}")
            logger.warning(f"  Error: {error_msg[:200]}")
            return False, error_msg
    
    def setup_models(self) -> Dict[str, bool]:
        """Process all model configs and attempt downloads."""
        logger.info("\n" + "="*70)
        logger.info("SETTING UP MODELS")
        logger.info("="*70)
        
        # Set cache directory to use local models folder
        cache_dir = str(self.models_dir)
        os.environ['HF_HOME'] = cache_dir
        
        results = {}
        config_files = sorted(self.config_dir.glob('apollo_*.json'))
        
        for config_file in config_files:
            config_name = config_file.stem
            logger.info(f"\n--- Processing: {config_name} ---")
            
            try:
                config = self.read_model_config(config_file)
                model_id = self.get_model_identifier(config)
                
                # Handle custom checkpoint paths
                if model_id.startswith('/') or model_id.startswith('ckpts'):
                    logger.warning(f"⚠ Custom checkpoint path (internal): {model_id}")
                    logger.warning(f"  This model requires the checkpoint file to be available locally")
                    results[config_name] = False
                else:
                    success, msg = self.download_model(model_id, cache_dir)
                    results[config_name] = success
                    if not success:
                        logger.warning(f"  Model may not be publicly available")
            
            except Exception as e:
                logger.error(f"Error processing {config_name}: {e}")
                results[config_name] = False
        
        return results
    
    def validate_setup(self) -> bool:
        """Validate that the environment is ready for training."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATING SETUP")
        logger.info("="*70)
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✓ Device: {device}")
        
        # Check data directory
        dataset_dir = self.root_dir / "Dataset"
        if dataset_dir.exists():
            logger.info(f"✓ Dataset directory found: {dataset_dir}")
            apollo_corpus = dataset_dir / "ApolloCorpus"
            if apollo_corpus.exists():
                pretrain_files = list((apollo_corpus / "pretrain").glob("*.json"))
                logger.info(f"  - ApolloCorpus pretrain files: {len(pretrain_files)}")
        else:
            logger.warning(f"⚠ Dataset directory not found: {dataset_dir}")
        
        # Check checkpoint directory
        checkpoint_dir = self.root_dir / "checkpoints_test"
        if checkpoint_dir.exists():
            logger.info(f"✓ Checkpoint directory found: {checkpoint_dir}")
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            logger.info(f"  - Checkpoint files: {len(checkpoints)}")
        
        return True
    
    def print_summary(self, model_results: Dict[str, bool]):
        """Print a summary of the setup."""
        logger.info("\n" + "="*70)
        logger.info("SETUP SUMMARY")
        logger.info("="*70)
        
        success_count = sum(1 for v in model_results.values() if v)
        total_count = len(model_results)
        
        logger.info(f"\nModels processed: {success_count}/{total_count}")
        for model_name, success in model_results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {model_name}")
        
        logger.info("\nNext steps:")
        logger.info("1. To train with default models:")
        logger.info("   python -m Gen_KD.train --dry-run")
        logger.info("\n2. To train with specific Apollo models:")
        logger.info("   python -m Gen_KD.train --models apollo-0.5B apollo-0.5B apollo-0.5B --dry-run")
        logger.info("\n3. To load custom checkpoint models:")
        logger.info("   Ensure checkpoint files are available in model_configs/")
        logger.info("   Then update --models arguments with local paths")
        logger.info("\nHuggingFace cache directory:")
        logger.info(f"   {os.environ.get('HF_HOME', '$HF_HOME')}")
    
    def run(self):
        """Execute the complete setup."""
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                logger.error("Setup failed: Missing dependencies")
                return False
            
            # Step 2: Setup models
            model_results = self.setup_models()
            
            # Step 3: Validate
            self.validate_setup()
            
            # Step 4: Print summary
            self.print_summary(model_results)
            
            return True
        
        except Exception as e:
            logger.error(f"Setup failed with error: {e}", exc_info=True)
            return False


def main():
    """Main entry point."""
    setup = LocalSetup()
    success = setup.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
