# Gen_KD - Teacher / Assistant / Student KD

This folder implements the KD recipe:

- Teacher: `Qwen/Qwen1.5-1.8B`, frozen
- Assistant: distilled Qwen 0.5B checkpoint, frozen
- Student: `HuggingFaceTB/SmolLM2-360M`, trainable
- Common projection dimension: `768`
- Teacher and assistant KD losses are averaged
- Total loss: `0.6 * loss_kd + 0.4 * loss_ce`

## Algorithm

```python
Teacher = Qwen1.5B          # frozen
Assistant = DistilledQwen0.5B # frozen
Student = SmolLM2_360M      # trainable

P_T = Linear(2048, 768)
P_A = Linear(1024, 768)
P_S = Linear(960, 768)

freeze(Teacher)
freeze(Assistant)

for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = input_ids.masked_fill(attention_mask == 0, -100)

    with torch.no_grad():
        H_T = Teacher(input_ids, output_hidden_states=True).hidden_states[-1]
        H_A = Assistant(input_ids, output_hidden_states=True).hidden_states[-1]

        Z_T = P_T(H_T)
        Z_A = P_A(H_A)

        p_T = mean_pool(Z_T, attention_mask)
        p_A = mean_pool(Z_A, attention_mask)

    out_S = Student(input_ids, labels=labels, output_hidden_states=True)
    H_S = out_S.hidden_states[-1]

    Z_S = P_S(H_S)
    p_S = mean_pool(Z_S, attention_mask)

    kd_teacher = MSE(p_S, p_T)
    kd_assistant = MSE(p_S, p_A)
    loss_kd = (kd_teacher + kd_assistant) / 2

    loss = 0.6 * loss_kd + 0.4 * out_S.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Project Structure

```text
Gen_KD/
|-- __init__.py
|-- config.py
|-- models.py
|-- projection.py
|-- trainer.py
|-- utils.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## Quick Start

```bash
pip install -r Gen_KD/requirements.txt
python -m Gen_KD.train
```

Pre-training verification:

```bash
python -m Gen_KD.verify_pipeline
```

To override models:

```bash
python -m Gen_KD.train --models "Qwen/Qwen1.5-1.8B" "./kd_pipeline/kd_checkpoints/Qwen_3/final.pt" "HuggingFaceTB/SmolLM2-360M"
```

The default dataset path is:

```text
./Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_90k.json
```

If that file is not present, the CLI falls back to the configured HuggingFace dataset.

## Key Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--models` | configured teacher/assistant/student trio | Model names or checkpoint paths |
| `--common-dim` | `768` | Shared projection dimension |
| `--lr` | `1e-5` | Learning rate |
| `--batch-size` | `2` | Training batch size |
| `--max-seq-len` | `512` | Tokenized sequence length |
| `--epochs` | `3` | Student training epochs |
| `--pooling` | `mean` | Pooling mode (`mean` or `cls`) |
| `--kd-loss-weight` | `0.6` | Weight for averaged KD loss |
| `--ce-loss-weight` | `0.4` | Weight for causal LM CE loss |
| `--dataset-path` | server Apollo JSON path | Local JSON dataset path |
| `--dry-run` | `false` | Uses 2 samples and 1 epoch for a quick check |

## Hidden-Dimension Check

The trainer validates the loaded model hidden sizes before creating projections:

```python
[2048, 1024, 960]
```

This ensures the projections are exactly:

```python
P_T = Linear(2048, 768)
P_A = Linear(1024, 768)
P_S = Linear(960, 768)
```
