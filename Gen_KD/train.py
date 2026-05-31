"""
train.py — CLI entry point for Generational Knowledge Distillation.

Usage:
    python -m Gen_KD.train                          # default config
    python -m Gen_KD.train --dry-run                # single batch per generation
    python -m Gen_KD.train --models gpt2 gpt2 gpt2  # specify models
"""

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from Gen_KD.config import GenKDConfig
from Gen_KD.models import ModelWrapper, load_tokenizer
from Gen_KD.trainer import GenKDTrainer
from Gen_KD.utils import setup_logging, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generational Knowledge Distillation")

    # Model
    p.add_argument(
        "--models", nargs="+", default=None,
        help="HuggingFace model names/paths. First = teacher, rest = students.",
    )

    # Training
    p.add_argument("--common-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Pooling & loss
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument(
        "--weight-strategy", choices=["uniform", "linear_decay"], default="uniform",
    )

    # Dataset
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-samples", type=int, default=None)

    # I/O
    p.add_argument("--device", default=None, help="auto-detect if not set")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-every", type=int, default=50)

    # Debug
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run only 1 batch per generation to verify correctness.",
    )

    return p.parse_args()


def build_config(args: argparse.Namespace) -> GenKDConfig:
    """Convert CLI args to a GenKDConfig."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = GenKDConfig(
        common_dim=args.common_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        epochs=1 if args.dry_run else args.epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        seed=args.seed,
        pooling_mode=args.pooling,
        weight_strategy=args.weight_strategy,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_text_field=args.text_field,
        max_samples=2 if args.dry_run else args.max_samples,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
    )

    if args.models:
        cfg.model_names = args.models

    return cfg


def build_dataloader(cfg: GenKDConfig, tokenizer) -> DataLoader:
    """Load and tokenize the dataset, return a DataLoader."""
    logger = setup_logging()
    logger.info(f"Loading dataset: {cfg.dataset_name} / {cfg.dataset_config}")

    ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
    )

    # Filter out empty strings
    ds = ds.filter(lambda x: len(x[cfg.dataset_text_field].strip()) > 0)

    # Optional sample cap
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))

    logger.info(f"Dataset size: {len(ds)} samples")

    def tokenize_fn(examples):
        return tokenizer(
            examples[cfg.dataset_text_field],
            max_length=cfg.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")

    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)


def main():
    args = parse_args()
    cfg = build_config(args)
    logger = setup_logging()

    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("  Generational Knowledge Distillation")
    logger.info("=" * 60)
    logger.info(f"  Models       : {cfg.model_names}")
    logger.info(f"  Generations  : {cfg.num_generations}")
    logger.info(f"  Device       : {cfg.device}")
    logger.info(f"  Common dim   : {cfg.common_dim}")
    logger.info(f"  Pooling      : {cfg.pooling_mode}")
    logger.info(f"  Weight strat : {cfg.weight_strategy}")
    if args.dry_run:
        logger.info("  *** DRY RUN — 1 batch per generation ***")
    logger.info("=" * 60)

    # ---- Load tokenizer (use the teacher's tokenizer for all models) ---- #
    tokenizer = load_tokenizer(cfg.model_names[0])

    # ---- Build dataloader ---- #
    dataloader = build_dataloader(cfg, tokenizer)

    # ---- Load models ---- #
    logger.info("Loading models...")
    models = []
    for i, name in enumerate(cfg.model_names):
        role = "Teacher" if i == 0 else f"Student {i}"
        logger.info(f"  [{role}] Loading {name}...")
        m = ModelWrapper(name, device=cfg.device)
        models.append(m)
    logger.info("All models loaded ✓")

    # ---- Train ---- #
    trainer = GenKDTrainer(config=cfg, models=models, dataloader=dataloader)
    trainer.train()

    logger.info("Done! 🎉")


if __name__ == "__main__":
    main()
