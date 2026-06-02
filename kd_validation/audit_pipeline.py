import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

# Add repository root to python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from Gen_KD.models import ModelWrapper, load_tokenizer
from Gen_KD.projection import ProjectionHead, pool
from Gen_KD.utils import set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"

def print_header(title):
    print("\n" + "="*80)
    print(f" {title.upper()}")
    print("="*80)

def main():
    student_ckpt_path = Path("kd_checkpoints/final.pt")
    assistant_ckpt_path = Path("kd_pipeline/kd_checkpoints/Qwen_3/final.pt")
    
    # -------------------------------------------------------------
    # 1. CHECKPOINT LOADING AUDIT
    # -------------------------------------------------------------
    print_header("1. Checkpoint Loading Audit")
    print(f"Student Checkpoint Path: {student_ckpt_path.resolve()}")
    print(f"Assistant Checkpoint Path: {assistant_ckpt_path.resolve()}")
    
    student_ckpt = torch.load(student_ckpt_path, map_location="cpu")
    assistant_ckpt = torch.load(assistant_ckpt_path, map_location="cpu")
    
    print("\nStudent Checkpoint Keys:", list(student_ckpt.keys()))
    print("Assistant Checkpoint Keys:", list(assistant_ckpt.keys()))
    
    print("\nMetadata inside student checkpoint:")
    if "metadata" in student_ckpt:
        for k, v in student_ckpt["metadata"].items():
            print(f"  {k}: {v}")
    else:
        print("  None")
        
    # Let's count parameters in the saved state dicts
    student_params = sum(p.numel() for p in student_ckpt["student_state_dict"].values())
    assistant_params = sum(p.numel() for p in assistant_ckpt["student_state_dict"].values())
    
    print(f"\nSaved Student Model parameters: {student_params:,}")
    print(f"Saved Assistant Model parameters: {assistant_params:,}")
    
    # Check projection state shapes in checkpoint
    print("\nSaved student projection weights shape:")
    for k, v in student_ckpt["proj_student_state_dict"].items():
        print(f"  {k}: {v.shape}")
        
    print("\nSaved assistant projection weights shape:")
    for k, v in assistant_ckpt["proj_student_state_dict"].items():
        print(f"  {k}: {v.shape}")

    # -------------------------------------------------------------
    # 2. PROJECTION CONSISTENCY AUDIT
    # -------------------------------------------------------------
    print_header("2. Projection Consistency Audit")
    
    # Replicate Training sequence of RNG to see what teacher projection weights were
    print("Simulating Training initialization sequence...")
    set_seed(42)
    # In training: first load teacher, assistant, student
    m_teacher = ModelWrapper("Qwen/Qwen1.5-1.8B", device="cpu")
    m_assistant = ModelWrapper(str(assistant_ckpt_path), device="cpu", base_model_name="Qwen/Qwen1.5-0.5B")
    m_student = ModelWrapper("HuggingFaceTB/SmolLM2-360M", device="cpu")
    
    # Then initialize projection heads in order: 0 (teacher), 1 (assistant), 2 (student)
    proj_teacher_train = ProjectionHead(2048, 768)
    proj_assistant_train = ProjectionHead(1024, 768)
    proj_student_train = ProjectionHead(960, 768)
    
    # Replicate Evaluation sequence of RNG
    print("Simulating Evaluation initialization sequence...")
    # In evaluation: load models first, then set seed, then initialize proj_teacher
    # we already loaded models above.
    set_seed(42)
    proj_teacher_eval = ProjectionHead(2048, 768)
    
    # Compare proj_teacher weights between training and evaluation simulation
    weight_match_teacher = torch.equal(proj_teacher_train.linear.weight, proj_teacher_eval.linear.weight)
    bias_match_teacher = torch.equal(proj_teacher_train.linear.bias, proj_teacher_eval.linear.bias)
    print(f"Teacher projection weight match: {weight_match_teacher}")
    print(f"Teacher projection bias match: {bias_match_teacher}")
    
    if not weight_match_teacher:
        diff_mean = (proj_teacher_train.linear.weight - proj_teacher_eval.linear.weight).abs().mean().item()
        print(f"Average absolute weight discrepancy: {diff_mean:.6f}")

    # Verify if evaluation loads student projection weights correctly
    proj_genkd_student = ProjectionHead(960, 768)
    proj_state = student_ckpt["proj_student_state_dict"]
    normalized_state = {}
    for k, v in proj_state.items():
        normalized_key = k.replace("projection.linear.", "linear.", 1)
        normalized_state[normalized_key] = v
    proj_genkd_student.load_state_dict(normalized_state)
    
    checkpoint_student_proj_weight = student_ckpt["proj_student_state_dict"]["linear.weight"]
    eval_student_proj_weight = proj_genkd_student.linear.weight
    print(f"Loaded student projection weights match checkpoint exactly: {torch.equal(checkpoint_student_proj_weight, eval_student_proj_weight)}")

    # -------------------------------------------------------------
    # 3. POOLING CONSISTENCY AUDIT
    # -------------------------------------------------------------
    print_header("3. Pooling Consistency Audit")
    
    # Check code of pool in Gen_KD.projection.pool
    import inspect
    print("Gen_KD pool code:")
    print(inspect.getsource(pool))
    
    # Compare with validation metrics pooling
    from kd_validation.validation_metrics import ValidationMetrics
    v_metrics = ValidationMetrics()
    print("ValidationMetrics._pool_hidden_states code:")
    print(inspect.getsource(v_metrics._pool_hidden_states))

    # -------------------------------------------------------------
    # 4. TOKENIZATION CONSISTENCY AUDIT
    # -------------------------------------------------------------
    print_header("4. Tokenization Consistency Audit")
    
    # Load tokenizers
    teacher_tok = load_tokenizer("Qwen/Qwen1.5-1.8B")
    assistant_tok_train = load_tokenizer("Qwen/Qwen1.5-0.5B")
    student_tok = load_tokenizer("HuggingFaceTB/SmolLM2-360M")
    
    sample_text = "Question: What is insulin? Answer: Insulin is a hormone that regulates blood glucose levels."
    
    encoded_teacher = teacher_tok(sample_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    encoded_assistant = assistant_tok_train(sample_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    encoded_student = student_tok(sample_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    
    print("Tokenizer comparison on sample text:")
    print(f"Teacher Tok: input_ids shape = {encoded_teacher['input_ids'].shape}, non-padded count = {encoded_teacher['attention_mask'].sum().item()}")
    print(f"Assistant Tok: input_ids shape = {encoded_assistant['input_ids'].shape}, non-padded count = {encoded_assistant['attention_mask'].sum().item()}")
    print(f"Student Tok: input_ids shape = {encoded_student['input_ids'].shape}, non-padded count = {encoded_student['attention_mask'].sum().item()}")
    
    # Check if Qwen1.5-1.8B and Qwen1.5-0.5B tokenizers are identical
    tokens_t = encoded_teacher['input_ids'][0]
    tokens_a = encoded_assistant['input_ids'][0]
    identical_qwen = torch.equal(tokens_t, tokens_a)
    print(f"Qwen 1.8B and Qwen 0.5B tokenized sequences identical: {identical_qwen}")

    # -------------------------------------------------------------
    # 5. HIDDEN STATE CONSISTENCY AUDIT
    # -------------------------------------------------------------
    print_header("5. Hidden State Consistency Audit")
    
    # Move models to GPU for actual forward pass checks
    m_teacher = ModelWrapper("Qwen/Qwen1.5-1.8B", device=device)
    m_assistant = ModelWrapper(str(assistant_ckpt_path), device=device, base_model_name="Qwen/Qwen1.5-0.5B")
    m_student = ModelWrapper("HuggingFaceTB/SmolLM2-360M", device=device)
    m_genkd_student = ModelWrapper(str(student_ckpt_path), device=device, base_model_name="HuggingFaceTB/SmolLM2-360M")
    
    ids_t = encoded_teacher["input_ids"].to(device)
    mask_t = encoded_teacher["attention_mask"].to(device)
    ids_a = encoded_assistant["input_ids"].to(device)
    mask_a = encoded_assistant["attention_mask"].to(device)
    ids_s = encoded_student["input_ids"].to(device)
    mask_s = encoded_student["attention_mask"].to(device)
    
    with torch.no_grad():
        out_t = m_teacher(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=True)
        out_a = m_assistant(input_ids=ids_a, attention_mask=mask_a, output_hidden_states=True)
        out_s = m_student(input_ids=ids_s, attention_mask=mask_s, output_hidden_states=True)
        out_gks = m_genkd_student(input_ids=ids_s, attention_mask=mask_s, output_hidden_states=True)
        
    print(f"Teacher hidden state shape: {out_t.hidden_states[-1].shape}")
    print(f"Assistant hidden state shape: {out_a.hidden_states[-1].shape}")
    print(f"Base Student hidden state shape: {out_s.hidden_states[-1].shape}")
    print(f"GenKD Student hidden state shape: {out_gks.hidden_states[-1].shape}")

    # -------------------------------------------------------------
    # 6. KD METRIC AUDIT (STEP-BY-STEP FOR SINGLE SAMPLE)
    # -------------------------------------------------------------
    print_header("6. KD Metric Audit")
    
    # Initialize projection layers used in validation
    proj_teacher_eval = proj_teacher_eval.to(device)
    proj_assistant_eval = ProjectionHead(1024, 768).to(device)
    # Load assistant projection
    proj_assistant_eval_state = assistant_ckpt["proj_student_state_dict"]
    norm_assistant_state = {}
    for k, v in proj_assistant_eval_state.items():
        norm_assistant_state[k.replace("projection.linear.", "linear.", 1)] = v
    proj_assistant_eval.load_state_dict(norm_assistant_state)
    
    proj_genkd_student = proj_genkd_student.to(device)
    
    # Also get the actual training teacher projection (proj_teacher_train)
    proj_teacher_train = proj_teacher_train.to(device)
    
    h_t = out_t.hidden_states[-1]
    h_a = out_a.hidden_states[-1]
    h_gks = out_gks.hidden_states[-1]
    
    # 6a. Using Training Teacher Projection (proj_teacher_train)
    z_t_train = proj_teacher_train(h_t)
    z_a = proj_assistant_eval(h_a)
    z_s_gks = proj_genkd_student(h_gks)
    
    p_t_train = pool(z_t_train, mask_t, "mean")
    p_a = pool(z_a, mask_a, "mean")
    p_s_gks = pool(z_s_gks, mask_s, "mean")
    
    kd_t_train = F.mse_loss(p_s_gks, p_t_train).item()
    kd_a_train = F.mse_loss(p_s_gks, p_a).item()
    
    # 6b. Using Evaluation Teacher Projection (proj_teacher_eval)
    z_t_eval = proj_teacher_eval(h_t)
    p_t_eval = pool(z_t_eval, mask_t, "mean")
    kd_t_eval = F.mse_loss(p_s_gks, p_t_eval).item()
    
    print("Single Sample KD loss comparison:")
    print(f"  KD_T with Training Teacher Projection:   {kd_t_train:.6f}")
    print(f"  KD_T with Evaluation Teacher Projection: {kd_t_eval:.6f}")
    print(f"  KD_A (Assistant):                        {kd_a_train:.6f}")
    
    print("\nPooled Vector Stats:")
    for name, p_vec in [("Teacher (Train Proj)", p_t_train), ("Teacher (Eval Proj)", p_t_eval), ("Assistant", p_a), ("GenKD Student", p_s_gks)]:
        print(f"  {name:22s}: mean={p_vec.mean().item():.6f}, std={p_vec.std().item():.6f}, min={p_vec.min().item():.6f}, max={p_vec.max().item():.6f}")

    # -------------------------------------------------------------
    # 7. COSINE SIMILARITY AUDIT
    # -------------------------------------------------------------
    print_header("7. Cosine Similarity Audit")
    
    cos_t_train = F.cosine_similarity(p_s_gks, p_t_train, dim=-1).mean().item()
    cos_t_eval = F.cosine_similarity(p_s_gks, p_t_eval, dim=-1).mean().item()
    cos_a = F.cosine_similarity(p_s_gks, p_a, dim=-1).mean().item()
    
    print(f"Cosine Similarity with Training Teacher Projection:   {cos_t_train:.6f}")
    print(f"Cosine Similarity with Evaluation Teacher Projection: {cos_t_eval:.6f}")
    print(f"Cosine Similarity with Assistant:                      {cos_a:.6f}")

    # -------------------------------------------------------------
    # 8. RECONCILIATION ON 100 SAMPLES
    # -------------------------------------------------------------
    print_header("8. Training vs Evaluation Reconciliation (100 Samples)")
    
    val_data_path = Path("Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa_2000_val.json")
    if not val_data_path.exists():
        print(f"Validation file {val_data_path} not found. Trying medicalGuideline_en_qa_validation.json...")
        val_data_path = Path("Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_validation.json")
        
    if not val_data_path.exists():
        print("No validation files found!")
        return
        
    with open(val_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:100]
        
    samples = []
    for item in data:
        if isinstance(item, list) and len(item) >= 2:
            samples.append(f"Question: {item[0]} Answer: {item[1]}")
        elif isinstance(item, dict):
            if "text" in item:
                samples.append(item["text"])
            elif "question" in item and "answer" in item:
                samples.append(f"Question: {item['question']} Answer: {item['answer']}")
                
    print(f"Loaded {len(samples)} samples for reconciliation.")
    
    total_kd_t_train = 0.0
    total_kd_t_eval = 0.0
    total_kd_a = 0.0
    
    for text in samples:
        enc_t = teacher_tok(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        enc_a = assistant_tok_train(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        enc_s = student_tok(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        
        id_t = enc_t["input_ids"].to(device)
        m_t = enc_t["attention_mask"].to(device)
        id_a = enc_a["input_ids"].to(device)
        m_a = enc_a["attention_mask"].to(device)
        id_s = enc_s["input_ids"].to(device)
        m_s = enc_s["attention_mask"].to(device)
        
        with torch.no_grad():
            o_t = m_teacher(input_ids=id_t, attention_mask=m_t, output_hidden_states=True)
            o_a = m_assistant(input_ids=id_a, attention_mask=m_a, output_hidden_states=True)
            o_gks = m_genkd_student(input_ids=id_s, attention_mask=m_s, output_hidden_states=True)
            
            p_t_tr = pool(proj_teacher_train(o_t.hidden_states[-1]), m_t, "mean")
            p_t_ev = pool(proj_teacher_eval(o_t.hidden_states[-1]), m_t, "mean")
            p_as = pool(proj_assistant_eval(o_a.hidden_states[-1]), m_a, "mean")
            p_st = pool(proj_genkd_student(o_gks.hidden_states[-1]), m_s, "mean")
            
            total_kd_t_train += F.mse_loss(p_st, p_t_tr).item()
            total_kd_t_eval += F.mse_loss(p_st, p_t_ev).item()
            total_kd_a += F.mse_loss(p_st, p_as).item()
            
    print(f"\nAverage metrics over {len(samples)} samples:")
    print(f"  KD_T (using Training Teacher Projection):   {total_kd_t_train / len(samples):.6f}")
    print(f"  KD_T (using Evaluation Teacher Projection): {total_kd_t_eval / len(samples):.6f}")
    print(f"  KD_A (Assistant):                           {total_kd_a / len(samples):.6f}")

if __name__ == "__main__":
    main()
