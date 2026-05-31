# Validation Pipeline - Complete Guide

## What This Pipeline Does

The validation pipeline provides comprehensive evaluation of your distilled Qwen student model by comparing it to:
1. **Base Student** - Fresh, untrained Qwen 0.5B model
2. **Teacher** - Frozen Qwen 1.8B model (reference)
3. **Distilled Student** - Your trained student model

## Three-Tier Validation

### 1. **Quantitative Validation** (Fast, ~5-10 min)
    
Computes numerical metrics on 1000 held-out validation samples:

**KD Loss (Knowledge Distillation Loss)**
```
Measures: How well does student match teacher hidden states?
Formula: MSE(proj_student_hidden, proj_teacher_hidden)
Expected: distilled_kd_loss < base_kd_loss
Why: Lower KD loss = student learned teacher's representations better
```

**CE Loss (Cross-Entropy Loss)**
```
Measures: How well does student predict next tokens?
Formula: Cross-entropy on next token prediction
Expected: distilled_ce_loss ≤ base_ce_loss
Why: Lower CE = better language modeling = better generation
```

**Perplexity**
```
Measures: How confident are predictions? (exp(CE_loss))
Expected: distilled_perplexity < base_perplexity
Why: Lower perplexity = model is more confident in good predictions
```

### 2. **Qualitative Validation** (Slow, ~15-30 min)

Generates answers on unseen medical questions:

```
Sample question: "What is treatment for condition X?"

Base Student:    [generates response A]
Distilled:       [generates response B]
Ground Truth:    [actual answer from dataset]

Check: Is B more medically accurate than A?
       Is B better structured?
       Is B more relevant?
```

### 3. **Comprehensive Comparison**

Creates side-by-side comparison:

| Metric | Base Student | Distilled | Better? |
|--------|-------------|-----------|---------|
| KD Loss | 0.523 | 0.412 | Distilled ✓ |
| CE Loss | 4.216 | 4.089 | Distilled ✓ |
| Perplexity | 67.2 | 59.5 | Distilled ✓ |

## Expected Results (Success Criteria)

### ✓ PASS: Successful Distillation
```
✓ KD Loss:  distilled_kd_loss < base_kd_loss
  Meaning:  Student learned to mimic teacher's representations
  
✓ CE Loss:  distilled_ce_loss ≤ base_ce_loss  
  Meaning:  Student is as good or better at text generation
```

### ✗ FAIL: Distillation Did Not Work
```
✗ KD Loss:  distilled_kd_loss >= base_kd_loss
  Problem:  Student didn't learn from teacher
  Fix:      Increase training epochs or learning rate
  
✗ CE Loss:  distilled_ce_loss > base_ce_loss
  Problem:  Student generation quality got worse
  Fix:      Reduce learning rate, use larger batch size
```

## How Validation Data is Created

1. **Full Dataset**: 99,685 medical Q&A pairs (medicalGuideline_en_qa.json)
2. **Training Set**: 10,000 samples (medicalGuideline_en_qa_10000.json)
3. **Validation Set**: 1,000 samples NOT in training (created by prepare_validation_data.py)

This ensures validation measures generalization, not memorization.

## Running the Validation

### Minimal (Recommended First Time)
```bash
cd /home/ror-technologies/Hikigai/Gen_KD
python3 kd_validation/quick_validate.py
```

### With Full Control
```bash
# Step 1: Create validation data (one time)
python3 kd_validation/prepare_validation_data.py

# Step 2: Run validation
python3 kd_validation/run_validation.py \
  --distilled-checkpoint kd_checkpoints/final.pt \
  --batch-size 4

# Step 3: Review results
cat kd_validation/results/validation_results.json
```

### With Qualitative (Slow but Detailed)
```bash
python3 kd_validation/run_validation.py \
  --distilled-checkpoint kd_checkpoints/final.pt \
  --with-qualitative
```

## Interpreting Results

### Great Results (What You Want)
```
KD Loss improved:    -11.2%  ✓ Student learned from teacher
CE Loss improved:     -3.0%  ✓ Generation quality improved  
Perplexity improved: -11.5%  ✓ Model is more confident
```
**Verdict**: ✓ **Distillation succeeded!**

### Good Results
```
KD Loss improved:     -5.0%  ✓ Student learned somewhat
CE Loss unchanged:     0.0%  = Generation quality same
Perplexity improved:  -2.0%  ✓ Slight confidence gain
```
**Verdict**: ✓ **Acceptable. More training may help.**

### Poor Results (Need to Fix)
```
KD Loss worsened:    +10.0%  ✗ Student didn't learn
CE Loss worsened:     +5.0%  ✗ Generation got worse
Perplexity worsened: +8.0%   ✗ Less confident
```
**Verdict**: ✗ **Distillation failed. Retrain with:**
- Higher learning rate (5e-5 to 1e-4)
- Fewer epochs initially (1-2)
- Larger batch size (4-8)

## Files Generated

```
kd_validation/
├── results/
│   └── validation_results.json      # Main results file
│       ├── quantitative/            # KD/CE losses
│       │   ├── base_student/
│       │   ├── distilled_student/
│       │   └── improvements/
│       └── qualitative/             # Sample generations
│           └── samples/
```

## Validation Metrics Explained

### KD Loss Deep Dive

```python
# What happens:
1. Teacher processes batch -> H_T (hidden states)
2. Student processes same batch -> H_S
3. Project both to common space: Z_T = proj_teacher(H_T), Z_S = proj_student(H_S)
4. Pool (mean with mask): p_T, p_S
5. KD_loss = MSE(p_S, p_T)

# If KD loss is lower for distilled:
   - Student representations are closer to teacher
   - Student learned the "teacher way" of thinking
   - This is the goal of knowledge distillation!
```

### CE Loss Deep Dive

```python
# What happens:
1. Model processes tokens: X -> logits
2. Compute CE loss on next token prediction
3. CE_loss = -log(P(y_true))

# If CE loss is lower for distilled:
   - Model makes better next token predictions
   - Better language understanding
   - Leads to higher quality generation
```

## Troubleshooting Validation

**Q: Validation dataset not found?**
```bash
python3 kd_validation/prepare_validation_data.py
```

**Q: Out of memory during validation?**
```bash
# Reduce batch size
python3 kd_validation/run_validation.py --batch-size 2
```

**Q: Checkpoint not found?**
```bash
# Check your checkpoint path
ls -la kd_checkpoints/
python3 kd_validation/run_validation.py --distilled-checkpoint <path>
```

**Q: Results look bad?**
```bash
# Review the distilled model:
python3 kd_pipeline/train_kd.py --epochs 5 --lr 5e-5
# Then revalidate
```

## Next Steps

1. **Run validation** to confirm distillation worked
2. **Review results** to understand performance
3. **Iterate training** if needed based on metrics
4. **Deploy distilled model** with confidence

---

For detailed implementation, see:
- [kd_validation/README.md](README.md) - API documentation
- [kd_validation/run_validation.py](run_validation.py) - Main validation script
- [kd_validation/validation_metrics.py](validation_metrics.py) - Metrics implementation
