#!/usr/bin/env python3
"""
evaluate_generational.py - Standalone evaluation pipeline for Generational KD experiment.
Compares:
1. Teacher: Qwen/Qwen1.5-1.8B
2. Assistant: Distilled Qwen0.5B (checkpoint)
3. Base Student: SmolLM2-360M
4. GenKD Student: SmolLM2-360M (trained checkpoint)

Computes KD Metrics, Cosine Similarity, Language Modeling metrics, and qualitative generations.
Saves CSV/JSON results and matplotlib plots.
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add repository root to python path to import Gen_KD modules
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from Gen_KD.models import ModelWrapper, load_tokenizer
from Gen_KD.projection import ProjectionHead, pool
from Gen_KD.utils import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GenKD_Eval")


class EvalDataset(Dataset):
    """Simple JSON dataset reader for evaluation."""
    def __init__(self, data_path: Path, tokenizers: list, max_seq_len: int, text_field: str = "text"):
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        logger.info(f"Loading evaluation dataset: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        self.samples = []
        # Support both QA list format [question, answer] and dict format
        for item in self.raw_data:
            if isinstance(item, list) and len(item) >= 2:
                self.samples.append(f"Question: {item[0]} Answer: {item[1]}")
            elif isinstance(item, dict):
                if text_field in item:
                    self.samples.append(str(item[text_field]))
                elif "question" in item and "answer" in item:
                    self.samples.append(f"Question: {item['question']} Answer: {item['answer']}")
                else:
                    # Fallback to concatenate all values
                    self.samples.append(" ".join(str(v) for v in item.values()))
            else:
                self.samples.append(str(item))

        logger.info(f"Dataset loaded. Total samples: {len(self.samples)}")
        self.tokenizers = tokenizers
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        features = {}
        for t_idx, tokenizer in enumerate(self.tokenizers):
            encoded = tokenizer(
                text,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            features[f"input_ids_{t_idx}"] = encoded["input_ids"].squeeze(0)
            features[f"attention_mask_{t_idx}"] = encoded["attention_mask"].squeeze(0)
        return features


def load_projection_head(checkpoint_path: Path, input_dim: int, common_dim: int, device: str) -> ProjectionHead:
    """Instantiate a ProjectionHead and load weights if available in checkpoint."""
    proj = ProjectionHead(input_dim, common_dim).to(device)
    if checkpoint_path.exists():
        logger.info(f"Loading projection from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        proj_state = (
            checkpoint.get("projection_state_dict")
            or checkpoint.get("proj_student_state_dict")
            or checkpoint.get("proj_teacher_state_dict")
        )
        if isinstance(proj_state, dict):
            # Normalize projection keys if they are nested
            normalized_state = {}
            for k, v in proj_state.items():
                normalized_key = k.replace("projection.linear.", "linear.", 1)
                normalized_state[normalized_key] = v
            proj.load_state_dict(normalized_state, strict=False)
            logger.info("  ✓ Successfully loaded projection weights")
        else:
            logger.warning("  ⚠ No projection state dict found in checkpoint. Using random init.")
    else:
        logger.warning(f"  ⚠ Checkpoint not found at {checkpoint_path}. Using random init.")
    proj.eval()
    return proj


def run_evaluation(args):
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizers
    logger.info("Loading tokenizers...")
    teacher_tok = load_tokenizer(args.teacher_model)
    student_tok = load_tokenizer(args.base_student_model)
    tokenizers = [teacher_tok, teacher_tok, student_tok] # teacher, assistant, student

    # 2. Build Dataloader
    try:
        eval_dataset = EvalDataset(Path(args.dataset_path), tokenizers, args.max_seq_len)
        dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 3. Load Models
    logger.info("Loading models...")
    
    # Teacher
    logger.info(f"Loading Teacher: {args.teacher_model}")
    teacher = ModelWrapper(args.teacher_model, device=device)
    teacher.eval()

    # Assistant
    logger.info(f"Loading Assistant from checkpoint: {args.assistant_checkpoint}")
    assistant = ModelWrapper(
        args.assistant_checkpoint, 
        device=device, 
        base_model_name="Qwen/Qwen1.5-0.5B"
    )
    assistant.eval()

    # Base Student
    logger.info(f"Loading Base Student: {args.base_student_model}")
    base_student = ModelWrapper(args.base_student_model, device=device)
    base_student.eval()

    # GenKD Student
    logger.info(f"Loading GenKD Student from checkpoint: {args.student_checkpoint}")
    genkd_student = ModelWrapper(
        args.student_checkpoint,
        device=device,
        base_model_name=args.base_student_model
    )
    genkd_student.eval()

    # 4. Load Projection Heads
    # Seed before projection initialization to keep random init consistent for teacher
    set_seed(args.seed)
    proj_teacher = ProjectionHead(2048, 768).to(device)
    proj_teacher.eval()

    proj_assistant = load_projection_head(Path(args.assistant_checkpoint), 1024, 768, device)
    proj_base_student = ProjectionHead(960, 768).to(device) # Base student uses default projection head
    proj_base_student.eval()

    proj_genkd_student = load_projection_head(Path(args.student_checkpoint), 960, 768, device)

    # 5. Evaluate Metrics
    logger.info("Evaluating models on validation dataset...")
    mse = nn.MSELoss()

    metrics = {
        "base": {"kd_t": 0.0, "kd_a": 0.0, "kd": 0.0, "cos_t": 0.0, "cos_a": 0.0, "ce": 0.0, "ppl": 0.0},
        "genkd": {"kd_t": 0.0, "kd_a": 0.0, "kd": 0.0, "cos_t": 0.0, "cos_a": 0.0, "ce": 0.0, "ppl": 0.0}
    }
    
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Inputs
            ids_t = batch["input_ids_0"].to(device)
            mask_t = batch["attention_mask_0"].to(device)
            ids_a = batch["input_ids_1"].to(device)
            mask_a = batch["attention_mask_1"].to(device)
            ids_s = batch["input_ids_2"].to(device)
            mask_s = batch["attention_mask_2"].to(device)

            # Causal LM Labels (mask padding)
            labels_s = ids_s.clone().masked_fill(mask_s == 0, -100)

            # Teacher representations
            out_t = teacher(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=True)
            p_t = pool(proj_teacher(out_t.hidden_states[-1]), mask_t, args.pooling)

            # Assistant representations
            out_a = assistant(input_ids=ids_a, attention_mask=mask_a, output_hidden_states=True)
            p_a = pool(proj_assistant(out_a.hidden_states[-1]), mask_a, args.pooling)

            # Base Student representations & LM loss
            out_bs = base_student(input_ids=ids_s, attention_mask=mask_s, labels=labels_s, output_hidden_states=True)
            p_bs = pool(proj_base_student(out_bs.hidden_states[-1]), mask_s, args.pooling)
            loss_ce_bs = out_bs.loss

            # GenKD Student representations & LM loss
            out_gks = genkd_student(input_ids=ids_s, attention_mask=mask_s, labels=labels_s, output_hidden_states=True)
            p_gks = pool(proj_genkd_student(out_gks.hidden_states[-1]), mask_s, args.pooling)
            loss_ce_gks = out_gks.loss

            # Base Student metric calculations
            kd_t_bs = mse(p_bs, p_t).item()
            kd_a_bs = mse(p_bs, p_a).item()
            cos_t_bs = F.cosine_similarity(p_bs, p_t, dim=-1).mean().item()
            cos_a_bs = F.cosine_similarity(p_bs, p_a, dim=-1).mean().item()

            metrics["base"]["kd_t"] += kd_t_bs
            metrics["base"]["kd_a"] += kd_a_bs
            metrics["base"]["kd"] += (kd_t_bs + kd_a_bs) / 2
            metrics["base"]["cos_t"] += cos_t_bs
            metrics["base"]["cos_a"] += cos_a_bs
            metrics["base"]["ce"] += loss_ce_bs.item()
            metrics["base"]["ppl"] += torch.exp(loss_ce_bs).item()

            # GenKD Student metric calculations
            kd_t_gks = mse(p_gks, p_t).item()
            kd_a_gks = mse(p_gks, p_a).item()
            cos_t_gks = F.cosine_similarity(p_gks, p_t, dim=-1).mean().item()
            cos_a_gks = F.cosine_similarity(p_gks, p_a, dim=-1).mean().item()

            metrics["genkd"]["kd_t"] += kd_t_gks
            metrics["genkd"]["kd_a"] += kd_a_gks
            metrics["genkd"]["kd"] += (kd_t_gks + kd_a_gks) / 2
            metrics["genkd"]["cos_t"] += cos_t_gks
            metrics["genkd"]["cos_a"] += cos_a_gks
            metrics["genkd"]["ce"] += loss_ce_gks.item()
            metrics["genkd"]["ppl"] += torch.exp(loss_ce_gks).item()

            num_batches += 1

    # Normalize metrics
    for model_key in metrics:
        for metric_key in metrics[model_key]:
            metrics[model_key][metric_key] /= num_batches

    logger.info("Quantitative evaluation complete.")
    logger.info(f"Base Student PPL: {metrics['base']['ppl']:.4f} | GenKD Student PPL: {metrics['genkd']['ppl']:.4f}")

    # 6. Qualitative Evaluation
    logger.info("Running qualitative evaluation...")
    medical_questions = [
        "What are the primary symptoms of Type 2 diabetes?",
        "How does hypertension affect cardiovascular health?",
        "What is the difference between viral and bacterial pneumonia?",
        "What are the common side effects of beta-blockers?",
        "How is asthma diagnosed and managed in adults?",
        "What causes rheumatoid arthritis and how is it treated?",
        "What are the risk factors for stroke?",
        "What is the role of insulin in regulating blood glucose?",
        "How can osteoporosis be prevented and treated?",
        "What are the symptoms and stages of Alzheimer's disease?",
        "What is the function of the thyroid gland and what is hypothyroidism?",
        "What are the primary treatment options for chronic kidney disease?",
        "How does atrial fibrillation increase stroke risk?",
        "What is coronary artery disease and how is it managed?",
        "What are the diagnostic criteria for major depressive disorder?",
        "What is the mechanism of action of penicillin?",
        "What are the long-term effects of chronic hepatitis B infection?",
        "How do statins help prevent cardiovascular events?",
        "What is gastroesophageal reflux disease (GERD) and how is it treated?",
        "What is the significance of HBA1c in monitoring diabetic patients?"
    ]

    qualitative_results = []
    
    def generate_response(model, tokenizer, prompt, max_new_tokens=100):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    for idx, question in enumerate(medical_questions, 1):
        logger.info(f"Generating answers for Q{idx}/20: {question[:40]}...")
        t_ans = generate_response(teacher, teacher_tok, question)
        a_ans = generate_response(assistant, teacher_tok, question)
        bs_ans = generate_response(base_student, student_tok, question)
        gks_ans = generate_response(genkd_student, student_tok, question)

        qualitative_results.append({
            "Question": question,
            "Teacher Answer": t_ans,
            "Assistant Answer": a_ans,
            "Base Student Answer": bs_ans,
            "GenKD Student Answer": gks_ans
        })

    # Save qualitative results
    qual_file = out_dir / "qualitative_eval.json"
    with open(qual_file, "w", encoding="utf-8") as f:
        json.dump(qualitative_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved qualitative results to {qual_file}")

    # 7. Generate Visualizations (Plots)
    logger.info("Generating comparison plots...")
    
    # KD Loss Comparison
    plt.figure(figsize=(7, 5), dpi=150)
    x = np.arange(3)
    width = 0.35
    base_kd = [metrics["base"]["kd_t"], metrics["base"]["kd_a"], metrics["base"]["kd"]]
    genkd_kd = [metrics["genkd"]["kd_t"], metrics["genkd"]["kd_a"], metrics["genkd"]["kd"]]
    plt.bar(x - width/2, base_kd, width, label="Base SmolLM2", color="#E06666")
    plt.bar(x + width/2, genkd_kd, width, label="GenKD SmolLM2", color="#6FA8DC")
    plt.xticks(x, ["KD_T (Teacher)", "KD_A (Assistant)", "KD (Combined)"])
    plt.ylabel("MSE Loss")
    plt.title("Knowledge Distillation Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kd_loss_comparison.png")
    plt.close()

    # Cosine Similarity Comparison
    plt.figure(figsize=(6, 5), dpi=150)
    x = np.arange(2)
    base_cos = [metrics["base"]["cos_t"], metrics["base"]["cos_a"]]
    genkd_cos = [metrics["genkd"]["cos_t"], metrics["genkd"]["cos_a"]]
    plt.bar(x - width/2, base_cos, width, label="Base SmolLM2", color="#E06666")
    plt.bar(x + width/2, genkd_cos, width, label="GenKD SmolLM2", color="#6FA8DC")
    plt.xticks(x, ["Teacher", "Assistant"])
    plt.ylabel("Cosine Similarity")
    plt.title("Representation Cosine Similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_similarity_comparison.png")
    plt.close()

    # CE Loss Comparison
    plt.figure(figsize=(5, 5), dpi=150)
    plt.bar(["Base SmolLM2", "GenKD SmolLM2"], [metrics["base"]["ce"], metrics["genkd"]["ce"]], color=["#E06666", "#6FA8DC"], width=0.5)
    plt.ylabel("Cross Entropy Loss")
    plt.title("Language Modeling Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ce_loss_comparison.png")
    plt.close()

    # Perplexity Comparison
    plt.figure(figsize=(5, 5), dpi=150)
    plt.bar(["Base SmolLM2", "GenKD SmolLM2"], [metrics["base"]["ppl"], metrics["genkd"]["ppl"]], color=["#E06666", "#6FA8DC"], width=0.5)
    plt.ylabel("Perplexity")
    plt.title("Model Perplexity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "perplexity_comparison.png")
    plt.close()

    # 8. Save CSV results
    csv_file = out_dir / "metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "KD_T", "KD_A", "KD", "Cosine_T", "Cosine_A", "CE_Loss", "Perplexity"])
        writer.writerow([
            "Base Student",
            f"{metrics['base']['kd_t']:.6f}",
            f"{metrics['base']['kd_a']:.6f}",
            f"{metrics['base']['kd']:.6f}",
            f"{metrics['base']['cos_t']:.6f}",
            f"{metrics['base']['cos_a']:.6f}",
            f"{metrics['base']['ce']:.6f}",
            f"{metrics['base']['ppl']:.6f}"
        ])
        writer.writerow([
            "GenKD Student",
            f"{metrics['genkd']['kd_t']:.6f}",
            f"{metrics['genkd']['kd_a']:.6f}",
            f"{metrics['genkd']['kd']:.6f}",
            f"{metrics['genkd']['cos_t']:.6f}",
            f"{metrics['genkd']['cos_a']:.6f}",
            f"{metrics['genkd']['ce']:.6f}",
            f"{metrics['genkd']['ppl']:.6f}"
        ])
    logger.info(f"Saved CSV results to {csv_file}")

    # 9. Generate final report.json
    def get_pct_imp(base, genkd, lower_is_better=True):
        if base == 0:
            return 0.0
        diff = (base - genkd) if lower_is_better else (genkd - base)
        return (diff / base) * 100

    report = {
        "best_model_by_KD": "GenKD Student" if metrics["genkd"]["kd"] < metrics["base"]["kd"] else "Base Student",
        "best_model_by_cosine_similarity": "GenKD Student" if (metrics["genkd"]["cos_t"] + metrics["genkd"]["cos_a"]) > (metrics["base"]["cos_t"] + metrics["base"]["cos_a"]) else "Base Student",
        "best_model_by_CE_loss": "GenKD Student" if metrics["genkd"]["ce"] < metrics["base"]["ce"] else "Base Student",
        "best_model_by_perplexity": "GenKD Student" if metrics["genkd"]["ppl"] < metrics["base"]["ppl"] else "Base Student",
        "percentage_improvements_GenKD_vs_Base": {
            "KD_T": get_pct_imp(metrics["base"]["kd_t"], metrics["genkd"]["kd_t"], True),
            "KD_A": get_pct_imp(metrics["base"]["kd_a"], metrics["genkd"]["kd_a"], True),
            "KD": get_pct_imp(metrics["base"]["kd"], metrics["genkd"]["kd"], True),
            "Cosine_T": get_pct_imp(metrics["base"]["cos_t"], metrics["genkd"]["cos_t"], False),
            "Cosine_A": get_pct_imp(metrics["base"]["cos_a"], metrics["genkd"]["cos_a"], False),
            "CE_Loss": get_pct_imp(metrics["base"]["ce"], metrics["genkd"]["ce"], True),
            "Perplexity": get_pct_imp(metrics["base"]["ppl"], metrics["genkd"]["ppl"], True)
        }
    }

    report_file = out_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report to {report_file}")
    
    logger.info("Evaluation complete! All artifacts generated successfully. 🎉")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Standalone Generational KD Evaluation")
    p.add_argument("--teacher-model", default="Qwen/Qwen1.5-1.8B")
    p.add_argument("--base-student-model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--assistant-checkpoint", default="./kd_pipeline/kd_checkpoints/Qwen_3/final.pt")
    p.add_argument("--student-checkpoint", default="./kd_checkpoints/final.pt", help="Trained GenKD student checkpoint path")
    p.add_argument("--dataset-path", default="./Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa_2000_val.json")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="evaluation_results")

    args = p.parse_args()
    run_evaluation(args)
