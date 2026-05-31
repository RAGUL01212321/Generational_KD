# Using Specific Dataset Files for KD Training

✅ **Updated to support specific dataset files!**

## Overview

You can now train with a specific dataset file instead of scanning a directory:

```bash
./train_kd --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json
```

## The Dataset File You Requested

**File:** `Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json`
- **Size:** 930.83 MB
- **Total Samples:** 524,855
- **Type:** Medical paper QA dataset

## Usage Examples

### Using the Specific File (Your Request)

```bash
cd /home/ror-technologies/Hikigai/Gen_KD

# Quick test with 10 samples
./train_kd \
  --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
  --max-samples 10 \
  --epochs 1

# Quick test with 1000 samples
./train_kd \
  --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
  --max-samples 1000 \
  --epochs 1

# Full training (all 524k samples)
./train_kd \
  --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
  --epochs 5 \
  --batch-size 8
```

### Using Directory Scanning (Default)

If you don't specify `--dataset-file`, it scans the directory:

```bash
# Uses all JSON files in Dataset/ApolloCorpus/pretrain/
./train_kd --epochs 5
```

### Custom Configuration

```bash
./train_kd \
  --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
  --max-samples 5000 \
  --epochs 3 \
  --batch-size 8 \
  --lr 5e-5 \
  --common-dim 768 \
  --seq-length 512
```

## How It Works

### Path Resolution

When you run from the root directory:
```bash
cd /home/ror-technologies/Hikigai/Gen_KD
./train_kd --dataset-file Dataset/...
```

The script:
1. Changes to `kd_pipeline/` folder
2. Checks if the path exists locally (it won't)
3. Automatically resolves it relative to parent (`../Dataset/...`)
4. Loads the file correctly

This means you always provide paths **as if you're in the Gen_KD root**, which is where you run commands from.

## Command Line Options

```bash
./train_kd --help
```

Shows all available options:

```
--dataset-file DATASET_FILE
    Specific dataset file (e.g., 
    Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json)

--max-samples MAX_SAMPLES
    Max samples to use from the file (default: None = all)

--epochs EPOCHS
    Number of training epochs

--batch-size BATCH_SIZE
    Batch size for training
```

## File Formats Supported

The data loader supports both:

1. **Single JSON file** (your request)
   - Can be a list of samples: `[{...}, {...}, ...]`
   - Or a single object: `{...}`

2. **Directory of JSON files** (default)
   - Scans all `.json` files in the directory
   - Loads each one

## Data Structure

The loader expects JSON with text data. It looks for:
- `text` field
- `content` field  
- Or falls back to string representation

For medicalPaper_en_qa.json, it will automatically extract the right data.

## Modified Files

### kd_data.py
- Added `_load_single_file()` method
- Updated `_load_data()` to handle both files and directories
- Detects if path is file or directory using `Path.is_file()`

### kd_config.py
- Added `dataset_file` parameter (Optional)
- Allows overriding `dataset_path` if set

### train_kd.py
- Added `--dataset-file` argument
- Automatic path resolution for files relative to parent directory
- Shows dataset path in training config output

## Example Training Session

```bash
# Start quick test
./train_kd \
  --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
  --max-samples 100 \
  --epochs 1 \
  --batch-size 4

# Output:
# ======================================================================
# KNOWLEDGE DISTILLATION TRAINING
# ======================================================================
# Teacher Model : FreedomIntelligence/Apollo-1.8B
# Student Model : FreedomIntelligence/Apollo-0.5B
# 
# Training Config:
#   Epochs       : 1
#   Batch size   : 4
#   Learning rate: 0.0001
#   Common dim   : 768
#   Seq length   : 512
#   Device       : cuda
#   Dataset      : ../Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json
#   Max samples  : 100
#   Max steps    : none
# ======================================================================
```

## Available Datasets

You can use any JSON file in the Dataset folder:

```
Dataset/ApolloCorpus/pretrain/
├── medicalBook_en_qa.json
├── medicalBook_en_text.json
├── medicalBook_zh_qa.json
├── medicalBook_zh_text.json
├── medicalGuideline_en_qa.json
├── medicalGuideline_en_text.json
├── medicalPaper_en_qa.json      ← Your requested file
├── medicalPaper_en_text.json
├── medicalPaper_es_qa.json
├── medicalPaper_es_text.json
├── medicalPaper_fr_qa.json
├── medicalPaper_fr_text.json
└── ...more files...
```

## Testing the Setup

```bash
# Test that path resolution works
cd /home/ror-technologies/Hikigai/Gen_KD
./train_kd --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
  --max-samples 10 --epochs 1
```

If you see this in output, you're good:
```
Dataset      : ../Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json
Loading data from ../Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json...
  Reading medicalPaper_en_qa.json... (10 total)
```

## Key Changes Summary

| Component | Change | Purpose |
|-----------|--------|---------|
| `kd_data.py` | Added file detection logic | Handle single files vs directories |
| `kd_config.py` | Added `dataset_file` field | Store specific file path |
| `train_kd.py` | Added `--dataset-file` arg | CLI interface for file selection |
| `train_kd.py` | Added path resolution | Fix relative paths from kd_pipeline |

## Tips

1. **For quick testing:**
   ```bash
   ./train_kd --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
     --max-samples 1000 --epochs 1
   ```

2. **For full training:**
   ```bash
   ./train_kd --dataset-file Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json \
     --epochs 5 --batch-size 8
   ```

3. **To use all files in a directory:**
   ```bash
   ./train_kd --epochs 5  # Uses all JSON in Dataset/ApolloCorpus/pretrain/
   ```

4. **Monitor training:**
   ```bash
   watch -n 5 'tail kd_pipeline/kd_checkpoints/metrics.json'
   ```

---

**Status**: ✅ **Ready to use specific dataset files!**

You can now use `Dataset/ApolloCorpus/pretrain/medicalPaper_en_qa.json` (524k+ samples) or any other JSON file in the dataset folder.
