"""
train.py — CLI entry point for Generational Knowledge Distillation.

Usage:
    python -m Gen_KD.train                          # default config
    python -m Gen_KD.train --dry-run                # single batch per generation
    python -m Gen_KD.train --models gpt2 gpt2 gpt2  # specify models
"""

import argparse
from pathlib import Path

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
    p.add_argument("--common-dim", type=int, default=768)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Pooling & loss
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--kd-loss-weight", type=float, default=0.6)
    p.add_argument("--ce-loss-weight", type=float, default=0.4)

    # Dataset
    p.add_argument(
        "--dataset-path",
        default="./Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json",
        help="Local JSON dataset path relative to the repo root; if present, this is loaded directly.",
    )
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-samples", type=int, default=None)

    # I/O
    p.add_argument("--device", default=None, help="auto-detect if not set")
    p.add_argument("--checkpoint-dir", default="kd_checkpoints")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--gradient-log-every", type=int, default=250)
    p.add_argument("--metrics-log-dir", default="logs")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--run-output-root", default=".")

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
        kd_loss_weight=args.kd_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_text_field=args.text_field,
        max_samples=2 if args.dry_run else args.max_samples,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        gradient_log_every=args.gradient_log_every,
        metrics_log_dir=args.metrics_log_dir,
        plots_dir=args.plots_dir,
        run_output_root=args.run_output_root,
    )

    if args.models:
        cfg.model_names = args.models

    return cfg


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [text for text in (_coerce_text(item).strip() for item in value) if text]
        return " ".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "prompt", "response", "instruction", "input", "output", "answer"):
            if key in value:
                text = _coerce_text(value[key]).strip()
                if text:
                    return text
        return " ".join(str(item) for item in value.values() if item is not None)
    return str(value)


def build_dataloader(cfg: GenKDConfig, tokenizers: list) -> DataLoader:
    """Load and tokenize the dataset, return a DataLoader."""
    logger = setup_logging()
    dataset_path = Path(cfg.dataset_path)

    if dataset_path.exists():
        logger.info(f"Loading local JSON dataset: {dataset_path}")
        ds = load_dataset("json", data_files=str(dataset_path), split=cfg.dataset_split)
    else:
        logger.info(f"Loading dataset: {cfg.dataset_name} / {cfg.dataset_config}")
        ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split=cfg.dataset_split,
        )

    # Filter out empty strings
    ds = ds.filter(lambda x: len(_coerce_text(x[cfg.dataset_text_field]).strip()) > 0)

    # Optional sample cap
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))

    logger.info(f"Dataset size: {len(ds)} samples")

    def tokenize_fn(examples):
        texts = [_coerce_text(value).strip() for value in examples[cfg.dataset_text_field]]
        features = {}
        for idx, tokenizer in enumerate(tokenizers):
            encoded = tokenizer(
                texts,
                max_length=cfg.max_seq_len,
                padding="max_length",
                truncation=True,
            )
            features[f"input_ids_{idx}"] = encoded["input_ids"]
            features[f"attention_mask_{idx}"] = encoded["attention_mask"]
        return features

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
    logger.info("  Roles        : teacher, assistant, student")
    logger.info(f"  Device       : {cfg.device}")
    logger.info(f"  Common dim   : {cfg.common_dim}")
    logger.info(f"  Pooling      : {cfg.pooling_mode}")
    logger.info(f"  Dataset path : {cfg.dataset_path}")
    logger.info(f"  Metrics dir  : {cfg.metrics_log_dir}")
    logger.info(f"  Plots dir    : {cfg.plots_dir}")
    logger.info(f"  Run root     : {cfg.run_output_root}")
    logger.info(
        "  Loss         : "
        f"kd={cfg.kd_loss_weight} * (0.8 * MSE(student, assistant) + 0.2 * MSE(student, teacher)) "
        f"+ ce={cfg.ce_loss_weight} * CE"
    )
    if args.dry_run:
        logger.info("  *** DRY RUN — 1 batch per generation ***")
    logger.info("=" * 60)

    # ---- Load models ---- #
    logger.info("Loading models...")
    models = []
    for i, name in enumerate(cfg.model_names):
        role = "Teacher" if i == 0 else "Assistant" if i == 1 else "Student"
        logger.info(f"  [{role}] Loading {name}...")
        m = ModelWrapper(name, device=cfg.device)
        models.append(m)
    logger.info("All models loaded")

    # ---- Load tokenizers ---- #
    # Qwen and SmolLM2 do not share a vocabulary, so tokenize the same text
    # separately for each model and compare only pooled projected embeddings.
    tokenizers = [load_tokenizer(model.model_name) for model in models]

    # ---- Build dataloader ---- #
    dataloader = build_dataloader(cfg, tokenizers)

    # ---- Train ---- #
    trainer = GenKDTrainer(config=cfg, models=models, dataloader=dataloader)
    trainer.train()

    logger.info("Done! 🎉")


if __name__ == "__main__":
    main()
