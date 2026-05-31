# Knowledge Distillation (KD) Pipeline

Distill Apollo 1.8B (Teacher) to Apollo 0.5B (Student) using hidden state-based knowledge distillation.

## Overview

### The Idea

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model by aligning their hidden representations.

**Architecture:**
```
Teacher (1.8B)  →  P_T (Projection)  →  Z_T (Common Space)  ↓
                                                              MSE Loss
Student (0.5B)  →  P_S (Projection)  →  Z_S (Common Space)  ↑
```

**Process:**
1. Extract hidden states from both teacher and student
2. Project them to a common dimension via learned projection layers
3. Pool (average) the projected representations
4. Calculate MSE loss and backprop through student only

## Pipeline

### Files

```
/home/ror-technologies/Hikigai/Gen_KD/
├── train_kd.py              # Main training script
├── kd_trainer.py            # Core trainer class
├── kd_config.py             # Configuration
├── kd_projection.py         # Projection layers
├── kd_data.py               # Data loading
├── kd_utils.py              # Utilities & evaluation
├── kd.sh                    # Quick launch script
└── KD_README.md             # This file
```

### Core Classes

**KDTrainer**
- Loads teacher (frozen) and student (trainable) models
- Creates projection layers
- Implements training loop
- Saves checkpoints

**KDConfig**
- All training hyperparameters
- Model IDs
- Data paths
- Device settings

**Projection Layers**
- `TeacherProjection`: Learned transformation for teacher
- `StudentProjection`: Learned transformation for student
- Both project to common dimension

## Training

### Quick Start

```bash
cd /home/ror-technologies/Hikigai/Gen_KD

# Default training (3 epochs)
python3 train_kd.py

# Or use the wrapper script
./kd.sh train

# Quick test (1000 samples, 1 epoch)
./kd.sh train-quick
```

### Full Training Command

```bash
python3 train_kd.py \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --common-dim 768 \
    --seq-length 512 \
    --save-every 100 \
    --device cuda
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--common-dim` | 768 | Projection dimension |
| `--seq-length` | 512 | Max sequence length |
| `--max-samples` | None | Limit dataset size |
| `--max-steps` | None | Stop after N steps |
| `--save-every` | 100 | Save checkpoint every N steps |
| `--device` | cuda | Device (cuda/cpu) |
| `--checkpoint-dir` | kd_checkpoints | Checkpoint directory |

### Resume Training

```bash
# From latest checkpoint
./kd.sh train-resume kd_checkpoints/step_100.pt

# With more epochs
./kd.sh train-resume kd_checkpoints/final.pt --epochs 5
```

## Training Loop (Pseudocode Implementation)

```python
for each batch x:
    # Forward pass
    with no_grad:
        H_T = T(x)              # Teacher hidden states
    H_S = S(x)                  # Student hidden states
    
    # Projection
    Z_T = P_T(H_T)              # Project to common dim
    Z_S = P_S(H_S)
    
    # Pooling
    p_T = mean(Z_T, axis=1)     # Mean over sequence
    p_S = mean(Z_S, axis=1)
    
    # Loss
    loss = MSE(p_S, p_T)
    
    # Backprop
    loss.backward()
    
    # Update (only student and projection)
    optimizer.step()
```

## Data

### Dataset Path

```
Dataset/ApolloCorpus/pretrain/
├── medicalBook_en_qa.json
├── medicalBook_en_text.json
├── medicalBook_zh_qa.json
├── medicalBook_zh_text.json
├── medicalGuideline_en_qa.json
├── medicalGuideline_en_text.json
├── medicalPaper_*.json
└── ...
```

### Loading

The data loader:
1. Scans dataset path for JSON files
2. Loads all files (or limited by `--max-samples`)
3. Tokenizes with student tokenizer
4. Pads/truncates to sequence length
5. Creates batches

### Limiting Data

For testing:
```bash
# Use only 1000 samples
./kd.sh train-quick

# Custom limit
python3 train_kd.py --max-samples 5000 --epochs 2
```

## Models

### Teacher (Frozen)

- **Model**: FreedomIntelligence/Apollo-1.8B
- **Hidden Dim**: 2560
- **Layers**: 32
- **Parameters**: 1.8B
- **Role**: Frozen, provides knowledge target
- **Requires Grad**: No

### Student (Trainable)

- **Model**: FreedomIntelligence/Apollo-0.5B
- **Hidden Dim**: 1024
- **Layers**: 24
- **Parameters**: 0.5B
- **Role**: Learns from teacher
- **Requires Grad**: Yes

### Projection Layers

- **Teacher Projection**: 2560 → 768 (default)
- **Student Projection**: 1024 → 768 (default)
- **Common Dimension**: 768 (configurable)

## Loss Function

**MSE Loss** on pooled representations:
```
L = ||mean(Z_S) - mean(Z_T)||²
```

Where:
- Z_S: Student projections (batch_size × common_dim)
- Z_T: Teacher projections (batch_size × common_dim)

## Checkpoints

### Saved Files

Each checkpoint contains:
- Student model weights
- Student projection weights
- Optimizer state
- Training config
- Training losses
- Global step count

### Locations

```
kd_checkpoints/
├── step_100.pt          # Every N steps
├── step_200.pt
├── final.pt             # Final checkpoint
├── interrupted.pt       # If interrupted
└── metrics.json         # Training metrics
```

### Usage

```bash
# List checkpoints
ls -lh kd_checkpoints/

# Resume from specific checkpoint
./kd.sh train-resume kd_checkpoints/step_500.pt

# Load in Python
from kd_trainer import KDTrainer
from kd_config import KDConfig

config = KDConfig()
trainer = KDTrainer(config)
trainer.load_checkpoint("kd_checkpoints/final.pt")
```

## Evaluation

### Using Distilled Student

After training, use the distilled student:

```python
from kd_utils import DistilledStudent

# Load distilled model
student = DistilledStudent(
    "FreedomIntelligence/Apollo-0.5B",
    checkpoint_path="kd_checkpoints/final.pt",
    device="cuda"
)

# Generate text
response = student.generate("What is AI?", max_length=100)
print(response)
```

### Compare Teacher vs Student

```python
from kd_utils import compare_models

compare_models(
    text="Explain machine learning",
    teacher_model_id="FreedomIntelligence/Apollo-1.8B",
    distilled_checkpoint="kd_checkpoints/final.pt"
)
```

### Metrics

Checkpoints save training metrics to `metrics.json`:

```json
{
  "train_losses": [0.45, 0.42, 0.40, ...],
  "global_step": 500,
  "timestamp": "2026-05-26T10:30:00"
}
```

## Performance Tips

### Faster Training

1. **Use GPU**: Default is CUDA, much faster than CPU
2. **Smaller dataset**: Use `--max-samples` for testing
3. **Larger batch size**: More GPU memory? Use `--batch-size 8`
4. **Fewer epochs**: Use `--epochs 1` for quick test

### Lower Memory

1. **Smaller batch**: `--batch-size 2`
2. **Smaller sequence**: `--seq-length 256`
3. **Smaller common-dim**: `--common-dim 512`
4. **CPU training**: `--device cpu`

### Better Quality

1. **More epochs**: `--epochs 5` or more
2. **More data**: Remove `--max-samples` limit
3. **Smaller learning rate**: `--lr 5e-5`
4. **Longer training**: More steps, more convergence

## Example Training Sessions

### Quick Test (5 min)

```bash
./kd.sh train-quick
# 1000 samples, 1 epoch, batch size 4
```

### Standard (30-60 min)

```bash
python3 train_kd.py --epochs 3 --batch-size 4
```

### Full (several hours)

```bash
python3 train_kd.py --epochs 5 --batch-size 8 --max-steps 5000
```

## Monitoring Training

Watch the log output:

```
Step   100 | Loss: 0.4523 | Avg Loss: 0.4512
Step   200 | Loss: 0.4201 | Avg Loss: 0.4357
Step   300 | Loss: 0.3945 | Avg Loss: 0.4223
...
```

- **Loss** should decrease over time
- **Avg Loss** should trend downward
- If loss plateaus, training is converging

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python3 train_kd.py --batch-size 2

# Reduce sequence length
python3 train_kd.py --seq-length 256

# Reduce common dimension
python3 train_kd.py --common-dim 512

# Use CPU
python3 train_kd.py --device cpu
```

### No GPU

```bash
# Use CPU explicitly
python3 train_kd.py --device cpu --batch-size 1
```

### Models Not Found

Ensure internet connection - models are downloaded from HuggingFace on first use.

### Slow Training

```bash
# Check if GPU is being used
nvidia-smi

# Use faster options
./kd.sh train-quick  # Test with small data
```

## File Descriptions

### train_kd.py
Main training entry point with argument parsing.

### kd_trainer.py
Core `KDTrainer` class that:
- Loads teacher and student models
- Creates projection layers
- Implements training step (pseudocode algorithm)
- Manages checkpoints
- Logs metrics

### kd_config.py
Configuration dataclass with all hyperparameters.

### kd_projection.py
Projection layer definitions:
- `ProjectionLayer`: Simple linear transformation
- `TeacherProjection`: For teacher
- `StudentProjection`: For student

### kd_data.py
Data loading:
- `ApolloKDDataset`: PyTorch dataset from JSON files
- `create_dataloader`: Creates DataLoader

### kd_utils.py
Utilities:
- `DistilledStudent`: Load and use distilled model
- `compare_models`: Compare teacher vs student
- `calculate_parameter_reduction`: Show efficiency gains

### kd.sh
Bash wrapper for quick training commands.

## Next Steps

1. **Run quick test**: `./kd.sh train-quick`
2. **Check results**: Look in `kd_checkpoints/metrics.json`
3. **Full training**: `python3 train_kd.py --epochs 5`
4. **Evaluate**: Use `kd_utils.py` to test distilled model
5. **Integrate**: Use distilled student in your pipeline

## Architecture Diagram

```
INPUT TEXT
    ↓
┌───────────────────┬───────────────────┐
│                   │                   │
▼                   ▼                   
TEACHER (1.8B)      STUDENT (0.5B)
│                   │
│ Extract Hidden    │ Extract Hidden
│ States (2560d)    │ States (1024d)
│                   │
▼                   ▼
P_T (2560→768d)     P_S (1024→768d)
│                   │
▼                   ▼
Z_T (768d)          Z_S (768d)
│                   │
└─────────┬─────────┘
          │
      Mean Pool
          │
    ┌─────┴─────┐
    ▼           ▼
  p_T (768d)   p_S (768d)
    │           │
    └─────┬─────┘
          │
       MSE Loss
          │
          ▼
    Backprop (S & P_S only)
```

## Key Insights

1. **Teacher frozen**: Teacher provides stable target
2. **Student trainable**: Student learns to match teacher
3. **Common space**: Forces representations to be similar
4. **Pooling**: Simplifies matching (sentence-level, not token-level)
5. **MSE loss**: Simple, effective for representation matching

## References

- Knowledge Distillation: original concept by Hinton et al.
- Hidden state matching: representation-based distillation
- Mean pooling: simplifies alignment while preserving information

---

**Status**: ✅ Complete and Ready

Created for distilling Apollo 1.8B → Apollo 0.5B with simple, clear implementation.
