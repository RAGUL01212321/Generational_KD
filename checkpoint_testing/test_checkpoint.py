#!/usr/bin/env python3
"""Validate and smoke-test KD checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
KD_PIPELINE = ROOT / "kd_pipeline"
if str(KD_PIPELINE) not in sys.path:
    sys.path.insert(0, str(KD_PIPELINE))

from kd_projection import StudentProjection  # noqa: E402


REQUIRED_KEYS = {
    "student_state_dict",
    "proj_student_state_dict",
    "config",
    "global_step",
}


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _checkpoint_summary(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    config = checkpoint.get("config", {})
    missing = REQUIRED_KEYS - set(checkpoint)

    print("\nCheckpoint")
    print("-" * 70)
    print(f"Path        : {checkpoint_path}")
    print(f"Size        : {_format_bytes(checkpoint_path.stat().st_size)}")
    print(f"Global step : {checkpoint.get('global_step', 'missing')}")
    print(f"Keys        : {', '.join(sorted(checkpoint.keys()))}")
    print(f"Missing     : {', '.join(sorted(missing)) if missing else 'none'}")
    print("\nConfig")
    print("-" * 70)
    for key in (
        "student_model_id",
        "teacher_model_id",
        "common_dim",
        "seq_length",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "optimizer",
    ):
        if key in config:
            print(f"{key:28}: {config[key]}")

    student_state = checkpoint.get("student_state_dict", {})
    proj_state = checkpoint.get("proj_student_state_dict", {})
    print("\nState Dicts")
    print("-" * 70)
    print(f"Student tensors    : {len(student_state):,}")
    print(f"Projection tensors : {len(proj_state):,}")


def _load_checkpoint(checkpoint_path: Path, map_location: str) -> dict[str, Any]:
    print(f"Loading checkpoint on {map_location}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    return checkpoint


def _load_student_and_projection(
    checkpoint: dict[str, Any],
    device: str,
    torch_dtype: torch.dtype,
):
    config = checkpoint["config"]
    student_model_id = config.get("student_model_id", "Qwen/Qwen1.5-0.5B")
    common_dim = int(config.get("common_dim", 768))

    print("\nLoading student model")
    print("-" * 70)
    print(f"Model       : {student_model_id}")
    print(f"Device      : {device}")
    print(f"Dtype       : {torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.load_state_dict(checkpoint["student_state_dict"], strict=True)
    model.to(device)
    model.eval()

    student_dim = model.config.hidden_size
    projection = StudentProjection(student_dim, common_dim)
    projection.load_state_dict(checkpoint["proj_student_state_dict"], strict=True)
    projection.to(device)
    projection.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        student_model_id,
        trust_remote_code=True,
    )

    print("Student and projection loaded successfully.")
    return model, projection, tokenizer


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> None:
    print("\nGeneration Smoke Test")
    print("-" * 70)
    print(f"Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nOutput")
    print("-" * 70)
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a KD checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for model loading/generation.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only load and print checkpoint metadata.",
    )
    parser.add_argument(
        "--load-model-only",
        action="store_true",
        help="Load checkpoint weights into the student model, but skip generation.",
    )
    parser.add_argument(
        "--prompt",
        default="What is diabetes?",
        help="Prompt for generation smoke test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Number of new tokens for generation.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for testing.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but not available.", file=sys.stderr)
        return 1

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    try:
        checkpoint = _load_checkpoint(checkpoint_path, map_location="cpu")
        _checkpoint_summary(checkpoint_path, checkpoint)

        missing = REQUIRED_KEYS - set(checkpoint)
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {sorted(missing)}")

        if args.metadata_only:
            print("\nMetadata check passed.")
            return 0

        model, _projection, tokenizer = _load_student_and_projection(
            checkpoint=checkpoint,
            device=args.device,
            torch_dtype=dtype_map[args.dtype],
        )

        if args.load_model_only:
            print("\nModel-load check passed.")
            return 0

        _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        print("\nCheckpoint test passed.")
        return 0
    except Exception as exc:
        print(f"\nCheckpoint test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
