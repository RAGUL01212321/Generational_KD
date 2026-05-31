# Gen_KD — Generational Knowledge Distillation

A PyTorch implementation of **Generational Knowledge Distillation**, where a frozen teacher model and N student models are trained **sequentially** — each student distills knowledge from **all** preceding models (teacher + earlier students).

## Algorithm Overview

```
For k = 1 to N:                             # each student generation
    Freeze all predecessors M[0..k-1]
    For each batch x:
        H_prev[i] = M[i](x)      (no grad)  # forward predecessors
        H_k       = M[k](x)      (grad)     # forward student

        p_prev[i] = pool(P[i](H_prev[i]))   # project + pool predecessors
        p_k       = pool(P[k](H_k))         # project + pool student

        loss = Σ w[k][i] · MSE(p_k, p_prev[i])
        loss.backward()
        update(M[k], P[k])                  # only update current student
```

## Project Structure

```
Gen_KD/
├── __init__.py        # Package init
├── config.py          # GenKDConfig dataclass
├── models.py          # ModelWrapper (HuggingFace model loader)
├── projection.py      # ProjectionHead + pooling utilities
├── trainer.py         # GenKDTrainer (core training loop)
├── utils.py           # Logging, seeds, checkpointing
├── train.py           # CLI entry point
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r Gen_KD/requirements.txt
```

### 2. Run a dry-run (tiny models, 1 batch)

```bash
cd /home/ragul/Desktop/Generational_distillation
python -m Gen_KD.train --dry-run
```

### 3. Run with custom models

```bash
python -m Gen_KD.train \
    --models gpt2 gpt2 gpt2 \
    --epochs 5 \
    --batch-size 16 \
    --lr 3e-5 \
    --common-dim 512
```

## CLI Arguments

| Argument            | Default                  | Description                              |
|---------------------|--------------------------|------------------------------------------|
| `--models`          | 3× `sshleifer/tiny-gpt2` | HF model names (teacher + students)     |
| `--common-dim`      | `256`                    | Shared projection dimension              |
| `--lr`              | `5e-5`                   | Learning rate                            |
| `--batch-size`      | `8`                      | Batch size                               |
| `--max-seq-len`     | `128`                    | Max sequence length                      |
| `--epochs`          | `3`                      | Epochs per generation                    |
| `--pooling`         | `mean`                   | Pooling mode (`mean` / `cls`)            |
| `--weight-strategy` | `uniform`                | Loss weight schedule                     |
| `--dataset`         | `wikitext`               | HuggingFace dataset name                 |
| `--dry-run`         | `false`                  | Run 1 batch per gen for testing          |

## Loss Weight Strategies

- **`uniform`**: Each predecessor contributes equally: `w[i] = 1/k`
- **`linear_decay`**: More recent predecessors get higher weight: `w[i] = (i+1) / Σ(1..k)`
