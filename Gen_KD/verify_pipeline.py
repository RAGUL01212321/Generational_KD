"""Pre-training verification for the Generational KD pipeline.

This script validates:
- model/tokenizer loading
- hidden-state shapes
- projection output shapes
- pooled representation shapes
- KD / CE loss computation
- numerical stability (NaN / Inf / large values)
- backward gradients and frozen-model behavior
- tiny optimizer-step training stability

Run from the Gen_KD package root, for example:

    python -m Gen_KD.verify_pipeline
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from datasets import load_dataset
from torch.optim import Adafactor

from Gen_KD.config import GenKDConfig
from Gen_KD.models import ModelWrapper, load_tokenizer
from Gen_KD.projection import ProjectionHead, pool
from Gen_KD.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-training verification for Gen_KD")
    parser.add_argument("--dataset-path", default=None, help="Override dataset JSON path")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--max-samples", type=int, default=10, help="Tiny subset size for optimizer checks")
    parser.add_argument("--optimizer-steps", type=int, default=5, help="Number of tiny optimizer steps to run")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Force device")
    parser.add_argument("--quiet", action="store_true", help="Reduce output noise")
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> GenKDConfig:
    cfg = GenKDConfig()
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.device is not None:
        cfg.device = args.device
    return cfg


def _assistant_base_model_id(cfg: GenKDConfig) -> str:
    # The assistant checkpoint is a distilled Qwen 0.5B checkpoint, so its
    # weights must be loaded into the original Qwen 1.5-0.5B architecture.
    return "Qwen/Qwen1.5-0.5B"


def _load_batch(cfg: GenKDConfig, tokenizer) -> Dict[str, torch.Tensor]:
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    if cfg.dataset_text_field not in ds.column_names:
        raise KeyError(
            f"Missing text field '{cfg.dataset_text_field}' in dataset columns: {ds.column_names}"
        )

    ds = ds.filter(lambda x: len(x[cfg.dataset_text_field].strip()) > 0)
    ds = ds.select(range(min(len(ds), 1)))

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
    return ds[0]


def _verify_tokenizer(tokenizer, name: str, text: str, max_seq_len: int) -> Tuple[bool, str, Dict[str, torch.Tensor]]:
    try:
        encoded = tokenizer(
            text,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            return False, f"{name} tokenizer produced invalid rank", encoded
        if input_ids.shape != attention_mask.shape:
            return False, f"{name} tokenizer produced mismatched shapes", encoded
        if input_ids.shape[1] != max_seq_len:
            return False, f"{name} tokenizer sequence length mismatch", encoded
        return True, f"{name} tokenizer OK", encoded
    except Exception as exc:
        return False, f"{name} tokenizer failed: {exc}", {}


def _nan_inf_large_check(name: str, tensor: torch.Tensor, threshold: float = 1e4) -> Tuple[bool, str]:
    if not torch.isfinite(tensor).all():
        return False, f"{name} contains NaN/Inf"
    if tensor.abs().max().item() > threshold:
        return False, f"{name} contains very large values (>{threshold})"
    return True, f"{name} is finite"


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().norm(2).item() ** 2)
    return math.sqrt(total)


def _get_transformer_blocks(model_wrapper: ModelWrapper):
    model = model_wrapper.model
    candidates = []
    for attr_name in ("model", "transformer", "base_model"):
        backbone = getattr(model, attr_name, None)
        if backbone is not None:
            candidates.append(backbone)
            for nested in ("layers", "h", "block", "blocks"):
                blocks = getattr(backbone, nested, None)
                if blocks is not None:
                    return blocks

    for nested in ("layers", "h", "block", "blocks"):
        blocks = getattr(model, nested, None)
        if blocks is not None:
            return blocks

    raise AttributeError(f"Could not locate transformer blocks for {model_wrapper.model_name}")


def _select_block(blocks, index: int):
    if len(blocks) == 0:
        raise ValueError("Empty block list")
    index = max(0, min(index, len(blocks) - 1))
    return blocks[index]


def main() -> int:
    args = parse_args()
    cfg = _resolve_config(args)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print("=" * 80)
    print("GEN_KD PRE-TRAINING VERIFICATION")
    print("=" * 80)
    print(f"Device            : {device}")
    print(f"Teacher           : {cfg.model_names[0]}")
    print(f"Assistant         : {cfg.model_names[1]}")
    print(f"Student           : {cfg.model_names[2]}")
    print(f"Common dim        : {cfg.common_dim}")
    print(f"Max seq length    : {cfg.max_seq_len}")
    print(f"Dataset path      : {cfg.dataset_path}")
    print(f"Tiny subset size  : {args.max_samples}")
    print(f"Optimizer steps   : {args.optimizer_steps}")
    print("=" * 80)

    results: Dict[str, bool] = {}

    try:
        teacher_tok = load_tokenizer(cfg.model_names[0])
        assistant_tok = load_tokenizer(cfg.model_names[1])
        student_tok = load_tokenizer(cfg.model_names[2])

        print("Tokenizer compatibility:")
        sample_text = "Pre-training verification for generational knowledge distillation."
        tok_checks = [
            _verify_tokenizer(teacher_tok, "Teacher", sample_text, cfg.max_seq_len),
            _verify_tokenizer(assistant_tok, "Assistant", sample_text, cfg.max_seq_len),
            _verify_tokenizer(student_tok, "Student", sample_text, cfg.max_seq_len),
        ]
        for ok, message, encoded in tok_checks:
            print(f"  {message}: {'PASS' if ok else 'FAIL'}")
            if encoded:
                print(
                    f"    input_ids shape={tuple(encoded['input_ids'].shape)} "
                    f"attention_mask shape={tuple(encoded['attention_mask'].shape)}"
                )
        if not all(ok for ok, _, _ in tok_checks):
            raise RuntimeError("Tokenizer compatibility check failed")

        batch = _load_batch(cfg, student_tok)
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
        labels = input_ids.clone().masked_fill(attention_mask == 0, -100)

        print("\nBatch shapes:")
        print(f"  input_ids      : {tuple(input_ids.shape)}")
        print(f"  attention_mask : {tuple(attention_mask.shape)}")
        print(f"  labels         : {tuple(labels.shape)}")

        teacher = ModelWrapper(cfg.model_names[0], device=cfg.device)
        assistant = ModelWrapper(
            cfg.model_names[1],
            device=cfg.device,
            base_model_name=_assistant_base_model_id(cfg),
        )
        student = ModelWrapper(cfg.model_names[2], device=cfg.device)

        teacher.hidden_dim = teacher.model.config.hidden_size
        assistant.hidden_dim = assistant.model.config.hidden_size
        student.hidden_dim = student.model.config.hidden_size

        print("\nHidden dimensions:")
        print(f"  Teacher  : {teacher.hidden_dim}")
        print(f"  Assistant: {assistant.hidden_dim}")
        print(f"  Student  : {student.hidden_dim}")

        expected_dims = cfg.expected_hidden_dims or [None, None, None]
        model_dim_checks = [
            ("teacher hidden dim", teacher.hidden_dim, expected_dims[0]),
            ("assistant hidden dim", assistant.hidden_dim, expected_dims[1]),
            ("student hidden dim", student.hidden_dim, expected_dims[2]),
        ]

        for label, actual, expected in model_dim_checks:
            ok = expected is None or actual == expected
            results[label] = ok
            print(f"  {label:22}: {'PASS' if ok else 'FAIL'} (actual={actual}, expected={expected})")

        teacher.freeze()
        assistant.freeze()
        student.unfreeze()

        proj_teacher = ProjectionHead(teacher.hidden_dim, cfg.common_dim).to(device)
        proj_assistant = ProjectionHead(assistant.hidden_dim, cfg.common_dim).to(device)
        proj_student = ProjectionHead(student.hidden_dim, cfg.common_dim).to(device)

        optimizer = Adafactor(
            list(student.parameters()) + list(proj_student.parameters()),
            lr=cfg.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
        )

        with torch.no_grad():
            teacher_out = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            assistant_out = assistant(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        student_out = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        h_teacher = teacher_out.hidden_states[-1]
        h_assistant = assistant_out.hidden_states[-1]
        h_student = student_out.hidden_states[-1]

        print("\nHidden-state shapes:")
        print(f"  Teacher   : {tuple(h_teacher.shape)}")
        print(f"  Assistant : {tuple(h_assistant.shape)}")
        print(f"  Student   : {tuple(h_student.shape)}")

        hidden_checks = [
            ("teacher hidden last dim", h_teacher.shape[-1], teacher.hidden_dim),
            ("assistant hidden last dim", h_assistant.shape[-1], assistant.hidden_dim),
            ("student hidden last dim", h_student.shape[-1], student.hidden_dim),
        ]
        for label, actual, expected in hidden_checks:
            ok = actual == expected
            results[label] = ok
            print(f"  {label:26}: {'PASS' if ok else 'FAIL'} (actual={actual}, expected={expected})")

        z_teacher = proj_teacher(h_teacher)
        z_assistant = proj_assistant(h_assistant)
        z_student = proj_student(h_student)

        print("\nProjection output shapes:")
        print(f"  Teacher   : {tuple(z_teacher.shape)}")
        print(f"  Assistant : {tuple(z_assistant.shape)}")
        print(f"  Student   : {tuple(z_student.shape)}")

        proj_checks = [
            ("teacher projection last dim", z_teacher.shape[-1], cfg.common_dim),
            ("assistant projection last dim", z_assistant.shape[-1], cfg.common_dim),
            ("student projection last dim", z_student.shape[-1], cfg.common_dim),
        ]
        for label, actual, expected in proj_checks:
            ok = actual == expected
            results[label] = ok
            print(f"  {label:28}: {'PASS' if ok else 'FAIL'} (actual={actual}, expected={expected})")

        p_teacher = pool(z_teacher, attention_mask, cfg.pooling_mode)
        p_assistant = pool(z_assistant, attention_mask, cfg.pooling_mode)
        p_student = pool(z_student, attention_mask, cfg.pooling_mode)

        print("\nPooled representation shapes:")
        print(f"  Teacher   : {tuple(p_teacher.shape)}")
        print(f"  Assistant : {tuple(p_assistant.shape)}")
        print(f"  Student   : {tuple(p_student.shape)}")

        pool_checks = [
            ("teacher pooled shape", tuple(p_teacher.shape), (input_ids.shape[0], cfg.common_dim)),
            ("assistant pooled shape", tuple(p_assistant.shape), (input_ids.shape[0], cfg.common_dim)),
            ("student pooled shape", tuple(p_student.shape), (input_ids.shape[0], cfg.common_dim)),
        ]
        for label, actual, expected in pool_checks:
            ok = actual == expected
            results[label] = ok
            print(f"  {label:22}: {'PASS' if ok else 'FAIL'} (actual={actual}, expected={expected})")

        kd_teacher = torch.nn.functional.mse_loss(p_student, p_teacher)
        kd_assistant = torch.nn.functional.mse_loss(p_student, p_assistant)
        loss_kd = (kd_teacher + kd_assistant) / 2
        loss_ce = student_out.loss if student_out.loss is not None else torch.tensor(0.0, device=device)
        total_loss = cfg.kd_loss_weight * loss_kd + cfg.ce_loss_weight * loss_ce

        print("\nLoss values:")
        print(f"  kd_teacher : {kd_teacher.item():.6f}")
        print(f"  kd_assistant: {kd_assistant.item():.6f}")
        print(f"  loss_kd    : {loss_kd.item():.6f}")
        print(f"  loss_ce    : {loss_ce.item():.6f}")
        print(f"  total_loss : {total_loss.item():.6f}")

        loss_checks = {
            "kd_teacher": kd_teacher,
            "kd_assistant": kd_assistant,
            "loss_kd": loss_kd,
            "loss_ce": loss_ce,
            "total_loss": total_loss,
        }
        for name, tensor in loss_checks.items():
            ok, message = _nan_inf_large_check(name, tensor)
            results[f"loss:{name}"] = ok
            print(f"  {name:12}: {'PASS' if ok else 'FAIL'} ({message})")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        print("\nGradient checks:")
        student_grad = any(param.grad is not None for param in student.parameters())
        student_proj_grad = any(param.grad is not None for param in proj_student.parameters())
        teacher_grad = any(param.grad is not None for param in teacher.parameters())
        assistant_grad = any(param.grad is not None for param in assistant.parameters())

        results["student grads"] = student_grad
        results["student projection grads"] = student_proj_grad
        results["teacher frozen"] = not teacher_grad
        results["assistant frozen"] = not assistant_grad

        print(f"  Student parameters receive gradients   : {'PASS' if student_grad else 'FAIL'}")
        print(f"  Student projection receives gradients  : {'PASS' if student_proj_grad else 'FAIL'}")
        print(f"  Teacher receives NO gradients          : {'PASS' if not teacher_grad else 'FAIL'}")
        print(f"  Assistant receives NO gradients        : {'PASS' if not assistant_grad else 'FAIL'}")

        print("\nGradient norms:")
        print(f"  Student projection : {_grad_norm(proj_student.parameters()):.6f}")

        try:
            teacher_blocks = _get_transformer_blocks(teacher)
            student_blocks = _get_transformer_blocks(student)
            teacher_first = _select_block(teacher_blocks, 0)
            teacher_last = _select_block(teacher_blocks, len(teacher_blocks) - 1)
            student_first = _select_block(student_blocks, 0)
            student_last = _select_block(student_blocks, len(student_blocks) - 1)
            print(f"  Teacher first block grad norm : {_grad_norm(teacher_first.parameters()):.6f}")
            print(f"  Teacher last block grad norm  : {_grad_norm(teacher_last.parameters()):.6f}")
            print(f"  Student first block grad norm  : {_grad_norm(student_first.parameters()):.6f}")
            print(f"  Student last block grad norm   : {_grad_norm(student_last.parameters()):.6f}")
        except Exception as exc:
            print(f"  Block grad norms unavailable: {exc}")

        print("\nTiny optimizer stability run:")
        optimizer.zero_grad(set_to_none=True)
        tiny_subset_steps = max(1, args.optimizer_steps)
        for step in range(tiny_subset_steps):
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                assistant_out = assistant(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )

            h_teacher = teacher_out.hidden_states[-1]
            h_assistant = assistant_out.hidden_states[-1]
            h_student = student_out.hidden_states[-1]

            p_teacher = pool(proj_teacher(h_teacher), attention_mask, cfg.pooling_mode)
            p_assistant = pool(proj_assistant(h_assistant), attention_mask, cfg.pooling_mode)
            p_student = pool(proj_student(h_student), attention_mask, cfg.pooling_mode)

            kd_teacher = torch.nn.functional.mse_loss(p_student, p_teacher)
            kd_assistant = torch.nn.functional.mse_loss(p_student, p_assistant)
            loss_kd = (kd_teacher + kd_assistant) / 2
            loss_ce = student_out.loss if student_out.loss is not None else torch.tensor(0.0, device=device)
            total_loss = cfg.kd_loss_weight * loss_kd + cfg.ce_loss_weight * loss_ce

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}")

            (total_loss / max(1, cfg.gradient_accumulation_steps)).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            print(
                f"  Step {step + 1:02d} | KD Loss: {loss_kd.item():.6f} | "
                f"CE Loss: {loss_ce.item():.6f} | Total Loss: {total_loss.item():.6f}"
            )

        print("\nFinal report:")
        shape_ok = all(results.get(key, False) for key in [
            "teacher hidden dim",
            "assistant hidden dim",
            "student hidden dim",
            "teacher hidden last dim",
            "assistant hidden last dim",
            "student hidden last dim",
            "teacher projection last dim",
            "assistant projection last dim",
            "student projection last dim",
            "teacher pooled shape",
            "assistant pooled shape",
            "student pooled shape",
        ])
        grad_ok = all(results.get(key, False) for key in [
            "student grads",
            "student projection grads",
            "teacher frozen",
            "assistant frozen",
        ])
        loss_ok = all(results.get(key, False) for key in [
            "loss:kd_teacher",
            "loss:kd_assistant",
            "loss:loss_kd",
            "loss:loss_ce",
            "loss:total_loss",
        ])

        print(f"  Shape checks            : {'PASS' if shape_ok else 'FAIL'}")
        print(f"  Gradient checks         : {'PASS' if grad_ok else 'FAIL'}")
        print(f"  Loss checks             : {'PASS' if loss_ok else 'FAIL'}")
        print("  Training stability      : PASS (tiny optimizer run completed)")
        overall = shape_ok and grad_ok and loss_ok
        print(f"\nOVERALL RESULT           : {'PASS' if overall else 'FAIL'}")
        return 0 if overall else 1

    except Exception as exc:
        print(f"\nOVERALL RESULT           : FAIL")
        print(f"Reason: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())