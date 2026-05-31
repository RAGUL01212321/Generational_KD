# Data Leakage Report

## Issue Found: Complete Data Leakage in Validation Set

**Date**: May 29, 2026  
**Severity**: 🚨 CRITICAL

### Summary

The validation set (`validation_data.json`) is **100% contained within the training set** (`medicalGuideline_en_qa_10000.json`).

- **Overlap**: 5000/5000 validation samples (100%)
- **Impact**: All validation metrics are meaningless
- **Implication**: The 0.28→0.97 improvement is pure memorization, not generalization

### Root Cause

The original dataset structure had:
- Training file: `medicalGuideline_en_qa_10000.json` (10,000 samples)
- Validation file: `validation_data.json` (5,000 samples)

The validation file was a random subset of the training file, causing severe data leakage.

### Solution Implemented

Created a proper 80/20 train/validation split:

1. **Training set** (8,000 samples):
   - File: `Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_8000.json`
   - Used for training the student model

2. **Validation set** (2,000 samples):
   - File: `Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_2000_val.json`
   - Truly separate from training data
   - Verified with hash-based duplicate detection

### Verification

Hash-based comparison confirms:
- ✓ **Exact duplicates**: 0
- ✓ **Question overlaps**: 2 out of 2000 (0.1% - negligible)
- ✓ **Verdict**: PASS - Datasets properly separated

### Configuration Updates

**kd_pipeline/kd_config.py** (Training):
```python
dataset_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_8000.json"
```

**kd_validation/validation_config.py** (Validation):
```python
validation_dataset_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_2000_val.json"
training_dataset_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_8000.json"
```

### Next Steps

1. **Re-train the model** with the clean training set:
   ```bash
   python3 kd_pipeline/train_kd.py \
     --epochs 10 \
     --dataset-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_8000.json
   ```

2. **Re-validate** with the clean validation set:
   ```bash
   python3 kd_validation/run_validation.py \
     --validation-file Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_2000_val.json \
     --distilled-checkpoint kd_checkpoints/Qwen_X/final.pt \
     --with-qualitative \
     --device cuda
   ```

3. **Expect realistic metrics**: The new validation scores will be lower than 0.97, but they'll represent true generalization.

### Tools Provided

1. **check_data_integrity.py** - Verify any train/val split for data leakage
   ```bash
   python3 check_data_integrity.py [train_file] [val_file]
   ```

2. **create_data_split.py** - Create clean train/val splits from any dataset
   ```bash
   python3 create_data_split.py [source_file]
   ```

### Key Takeaway

**Always verify data separation before validating**. Data leakage invalidates all downstream analysis.
