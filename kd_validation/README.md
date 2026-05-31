# Knowledge Distillation Validation Pipeline

Complete validation pipeline for evaluating distilled student models against base students.

## Overview

This validation pipeline provides:

1. **Quantitative Metrics**: KD loss and CE loss on held-out validation data
2. **Qualitative Evaluation**: Generate and compare outputs from base and distilled students  
3. **Comprehensive Reporting**: JSON results with all metrics and improvements

## Quick Start

### 1. Create Validation Dataset

```bash
python3 kd_validation/prepare_validation_data.py \
  --source-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa.json \
  --training-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json \
  --output-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_validation.json \
  --validation-size 1000
```

This creates a validation set of 1000 samples that were NOT in the training data.

### 2. Run Validation (Quantitative Only - Fast)

```bash
python3 kd_validation/run_validation.py \
  --validation-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_validation.json \
  --distilled-checkpoint kd_checkpoints/final.pt \
  --device cuda \
  --batch-size 4
```

**Expected output:**
```
Base Student:
  KD Loss:     0.5234
  CE Loss:     4.2156
  Perplexity:  67.23

Distilled Student:
  KD Loss:     0.4123  ✓ (lower = learned teacher better)
  CE Loss:     4.0892  ✓ (lower = better generation)

Improvements:
  KD Loss:  -11.2%  ✓ PASS
  CE Loss:  -3.0%   ✓ PASS
```

### 3. Run with Qualitative Evaluation (Slow)

```bash
python3 kd_validation/run_validation.py \
  --validation-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_validation.json \
  --distilled-checkpoint kd_checkpoints/final.pt \
  --with-qualitative
```

This compares generations from base and distilled students on validation samples.

### 4. Quick One-Command Validation

```bash
python3 kd_validation/quick_validate.py
```

This runs the full pipeline:
1. Creates validation dataset if needed
2. Runs quantitative validation
3. Saves all results

## Metrics Explained

### Quantitative Metrics

**KD Loss (Knowledge Distillation Loss)**
- Measures how well student hidden states match teacher hidden states
- Lower is better (distilled student should be closer to teacher)
- Expected: `distilled_kd_loss < base_kd_loss`

**CE Loss (Cross-Entropy Loss)**
- Language modeling loss on next token prediction
- Lower is better (better text generation)
- Expected: `distilled_ce_loss ≤ base_ce_loss`

**Perplexity**
- Exponential of CE loss
- Lower is better (more confident predictions)
- Related to CE loss

### Expected Results

For successful distillation:
1. **KD Loss Test**: ✓ PASS - Distilled student learns teacher representations
2. **CE Loss Test**: ✓ PASS - Distilled student has better generation quality

If both tests pass: **Distillation was successful!**

## File Structure

```
kd_validation/
├── __init__.py                      # Package init
├── validation_config.py             # Validation configuration
├── prepare_validation_data.py       # Create held-out validation set
├── validation_metrics.py            # Compute KD/CE losses
├── qualitative_eval.py              # Generate and compare outputs
├── run_validation.py                # Main validation script
├── quick_validate.py                # One-command validation
├── README.md                        # This file
└── results/
    └── validation_results.json      # Results (auto-generated)
```

## Results File

Results are saved to `kd_validation/results/validation_results.json`:

```json
{
  "quantitative": {
    "base_student": {
      "kd_loss": 0.5234,
      "ce_loss": 4.2156,
      "perplexity": 67.23
    },
    "distilled_student": {
      "kd_loss": 0.4123,
      "ce_loss": 4.0892,
      "perplexity": 59.54
    },
    "improvements": {
      "kd_loss_diff": 0.1111,
      "kd_loss_improvement_pct": -11.2,
      "ce_loss_diff": 0.1264,
      "ce_loss_improvement_pct": -3.0
    },
    "tests_passed": {
      "kd_loss_test": true,
      "ce_loss_test": true
    }
  },
  "qualitative": {
    "num_samples": 20,
    "samples": [
      {
        "question": "...",
        "base_student": "...",
        "distilled_student": "..."
      }
    ]
  }
}
```

## Validation Workflow

1. **Prepare Data**: Create held-out validation set (not in training)
2. **Load Models**: 
   - Teacher (frozen, reference)
   - Base student (fresh, untrained reference)
   - Distilled student (trained via KD)
3. **Compute Metrics**:
   - KD loss: How close to teacher?
   - CE loss: How good at generation?
4. **Compare**: 
   - Base vs Distilled: Did distillation help?
5. **Report**: Save results with interpretation

## Tips

- **Batch Size**: Use 2-4 for large medical texts
- **Validation Size**: 500-1000 samples is good
- **Time**: Quantitative validation takes ~5-10 min, qualitative adds 5-15 min
- **Device**: Requires GPU for reasonable speed

## Troubleshooting

**Out of Memory**: Reduce batch size or validation size
```bash
python3 kd_validation/run_validation.py --batch-size 2
```

**Validation data not found**: Create it first
```bash
python3 kd_validation/prepare_validation_data.py
```

**Checkpoint not found**: Check path to distilled model checkpoint
```bash
python3 kd_validation/run_validation.py --distilled-checkpoint <path-to-checkpoint>
```
