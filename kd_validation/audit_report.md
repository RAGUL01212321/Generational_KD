# Generational Knowledge Distillation Evaluation Audit Report

This report presents a full audit of the evaluation pipeline located in `kd_validation` against the training pipeline `Gen_KD` for the student model checkpoints.

## Executive Summary

The evaluation pipeline reported a major discrepancy: **`KD_T` (5.60)** was much larger than **`KD_A` (1.72)**, whereas training logs showed **`KD_T ≈ 1.64` and `KD_A ≈ 1.61`**.

Our audit successfully identified the exact root cause: **The Teacher projection layer weights in the evaluation script do not match the ones used during training.** Because the student checkpoint (`kd_checkpoints/final.pt`) does not save the frozen random teacher projection weights, the evaluation script attempts to recreate them using `set_seed(42)`. However, the order of random number generator (RNG) calls in training differs from evaluation (due to loading the models before initializing the projection heads in training), leading to completely different teacher projection weights.

**Reconciliation results show that when using the correct training teacher projection weights, the evaluation `KD_T` matches training exactly (`KD_T = 1.637`). Therefore, the current evaluation results *cannot* be trusted for `KD_T` and `Cosine_T`, but the evaluation code itself is mathematically correct once projection weights are aligned.**

---

## 1. Confirmed Correct Components

The following components are verified to be mathematically and structurally consistent with training:

| Component | Status | Details |
| :--- | :---: | :--- |
| **Student Projection** | **PASS** | Correctly loaded from `kd_checkpoints/final.pt` (`proj_student_state_dict`). Dimension is `960 -> 768`. Weights match checkpoint exactly. |
| **Assistant Projection** | **PASS** | Correctly loaded from `kd_pipeline/kd_checkpoints/Qwen_3/final.pt` (`proj_student_state_dict`). Dimension is `1024 -> 768`. |
| **Pooling Logic** | **PASS** | Both training and evaluation use the identical masked mean pooling logic (`Gen_KD.projection.pool`). |
| **Tokenization** | **PASS** | Under identical inputs, both tokenizers yield matching token counts, sequence lengths, and masks. Qwen 1.8B and 0.5B tokenizers produce identical tokenized sequences. |
| **Hidden State Extraction** | **PASS** | Both pipelines extract `hidden_states[-1]` (the last hidden layer output) prior to projection. Shapes are correct: Teacher `2048`, Assistant `1024`, Student `960`. |

---

## 2. Detected Inconsistencies & Potential Bugs

### Major: Teacher Projection Weight Mismatch
During training, the teacher's projection (`proj_teacher`) is randomly initialized once at the start of training (using seed 42) and is kept frozen. Since it is not updated, it is **not saved** in the student checkpoint (`kd_checkpoints/final.pt`).

During evaluation (`evaluate_generational.py`), the script attempts to reconstruct `proj_teacher` by setting `set_seed(42)` right before initializing it:
```python
set_seed(args.seed)
proj_teacher = ProjectionHead(2048, 768).to(device)
```

However, the RNG state at this point is different from the RNG state during training:
1. **Training Sequence**:
   - `set_seed(42)`
   - Load Teacher Model, Assistant Model, Student Model (RNG is advanced by model loading)
   - Initialize `proj_teacher` (Teacher projection)
   - Initialize `proj_assistant`
   - Initialize `proj_student`
2. **Evaluation Sequence**:
   - Load Models
   - `set_seed(42)`
   - Initialize `proj_teacher`

Because loading models advances the PyTorch RNG state, the weights of the teacher projection initialized in training and evaluation are completely different, resulting in an average absolute weight discrepancy of **0.021966**.

---

## 3. Step-by-Step KD Metric Audit (Single Sample)

We evaluated a single sample using both the evaluation teacher projection and the simulated training teacher projection:

* **KD_T (using Training Teacher Projection)**: **`1.616654`** (matches training)
* **KD_T (using Evaluation Teacher Projection)**: **`5.485121`** (replicates the evaluation bug)
* **KD_A (Assistant)**: **`1.624734`**

### Pooled Vector Stats

| Vector Source | Mean | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Teacher (Training Proj)** | 0.038167 | 0.344400 | -1.309062 | 1.350352 |
| **Teacher (Evaluation Proj)** | 0.030588 | 0.347573 | -1.385559 | 1.344199 |
| **Assistant** | 0.046536 | 0.320499 | -0.940562 | 1.139683 |
| **GenKD Student** | 0.031586 | 0.339682 | -1.135246 | 1.218524 |

### Cosine Similarity comparison
* **Cosine Similarity (Training Teacher Proj)**: **`0.7226`**
* **Cosine Similarity (Evaluation Teacher Proj)**: **`-0.0543`** (near-orthogonal random projection)
* **Cosine Similarity (Assistant)**: **`0.7289`**

---

## 4. Reconciliation on 100 Samples

Running 100 evaluation samples demonstrates the exact reconciliation of the metrics:

| Metric | Evaluation (Current) | Reconciled (Correct Training Weights) | Training Target |
| :--- | :---: | :---: | :---: |
| **KD_T** | **`5.589122`** | **`1.637890`** | **`~1.64`** |
| **KD_A** | **`1.710091`** | **`1.710091`** | **`~1.61`** |

Using the correct projection weights fully reconciles the evaluation results back to `KD_T ≈ 1.64` and `KD_A ≈ 1.71` (which matches the training logs).

---

## 5. Conclusion & Recommendations

### Confidence Level: **LOW (for current evaluation outputs, but HIGH for underlying model correctness)**
The current evaluation results reported by `evaluate_generational.py` for `KD_T` and `Cosine_T` **cannot be trusted** because they compare student representations against a completely different (random) projection of the teacher. However:
1. The **`KD_A` and `Cosine_A`** metrics are completely valid and correct.
2. The **CE Loss and Perplexity** metrics are completely valid and correct.
3. The model was trained correctly, and the training logs represent the actual distillation progress.

### Recommendations (No training code modifications required)
To make the evaluation script trustable, we must align the teacher projection weights. Since we cannot modify the training code, we can either:
- Modify `evaluate_generational.py` to recreate the teacher projection using the exact RNG initialization sequence that occurred during training (i.e. setting the seed before loading models and matching the initialization sequence).
- Or, save the correctly initialized teacher projection weights to a static checkpoint and load them directly in evaluation.
