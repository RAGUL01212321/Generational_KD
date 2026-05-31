# Knowledge Distillation Flow

This document describes the current KD training flow in this repo.

## 1. Inputs

Training starts from:

- Teacher model: `Qwen/Qwen1.5-1.8B`
- Student model: `Qwen/Qwen1.5-0.5B`
- Dataset JSON file, for example:
  `Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json`
- Tokenizer from the student model

The current recommended run shape is:

```text
batch_size = 2
gradient_accumulation_steps = 4
effective_batch_size = 8
```

## 2. Model Loading

The trainer loads two models:

```text
Teacher model
  -> loaded frozen
  -> moved to GPU
  -> eval mode
  -> no gradients

Student model
  -> loaded trainable
  -> moved to GPU
  -> train mode
  -> gradients enabled
```

Projection layers are also created:

```text
Teacher hidden size -> common_dim
Student hidden size -> common_dim
```

The teacher projection is frozen. The student projection is trained.

## 3. Dataset Flow

The dataset loader reads samples from the selected JSON file.

For each sample:

```text
raw text
  -> student tokenizer
  -> input_ids
  -> attention_mask
```

Each batch is streamed from the dataloader. The code no longer stores a full epoch of batches in memory.

## 4. One Training Microbatch

For each microbatch:

```text
CPU batch tensors
  -> move to GPU
```

The batch contains:

```text
input_ids
attention_mask
```

## 5. Teacher Forward

The teacher runs under `torch.no_grad()`.

```text
input_ids + attention_mask
  -> teacher backbone
  -> teacher final hidden states
```

Important details:

- The teacher does not compute gradients.
- The teacher does not compute LM logits for KD.
- Only final hidden states are used.

Then:

```text
teacher hidden states
  -> teacher projection
  -> projected teacher states
  -> mean pool over real tokens
  -> p_T
```

`p_T` is the teacher target representation.

## 6. Student Forward

The student backbone runs with gradients enabled.

```text
input_ids + attention_mask
  -> student backbone
  -> student final hidden states
```

Then the student hidden states are used in two ways.

First, for KD:

```text
student hidden states
  -> student projection
  -> projected student states
  -> mean pool over real tokens
  -> p_S
```

Second, for language modeling:

```text
student hidden states
  -> student LM head
  -> logits
```

## 7. Losses

The current objective combines two losses.

### Hidden KD Loss

This aligns student representations with teacher representations:

```text
kd_loss = MSE(p_S, p_T)
```

### Causal LM Loss

This keeps the student good at next-token prediction:

```text
ce_loss = CrossEntropy(student_logits, shifted_input_ids)
```

Padding tokens are ignored using the attention mask.

### Total Loss

The final loss is:

```text
total_loss =
  kd_loss_weight * kd_loss
  + ce_loss_weight * ce_loss
```

Current defaults:

```text
kd_loss_weight = 0.3
ce_loss_weight = 0.7
```

This means generation quality is prioritized while still distilling teacher representations.

## 8. Backpropagation

The trainer backpropagates through:

```text
student model
student projection
```

It does not backpropagate through:

```text
teacher model
teacher projection
```

With gradient accumulation:

```text
microbatch 1 -> backward, no optimizer step
microbatch 2 -> backward, no optimizer step
microbatch 3 -> backward, no optimizer step
microbatch 4 -> backward, optimizer step
```

For `batch_size=2` and `gradient_accumulation_steps=4`:

```text
2 samples per microbatch
4 microbatches per optimizer step
8 effective samples per optimizer update
```

## 9. Optimizer Step

The optimizer is `Adafactor`.

After enough microbatches are accumulated:

```text
clip gradients
  -> optimizer.step()
  -> optimizer.zero_grad()
```

Adafactor is used because it requires less optimizer memory than AdamW.

## 10. Logging

Every `log_every` steps, the trainer prints:

```text
Step
Total loss
KD loss
CE loss
Average loss
```

Example:

```text
Step  2501 | Loss: 1.2345 | KD: 0.2100 | CE: 1.6700 | Avg Loss: 1.5000
```

Training logs are also written under:

```text
kd_pipeline/kd_checkpoints/logs/
```

## 11. Checkpointing

Checkpoints save:

```text
global_step
student_state_dict
proj_student_state_dict
optimizer_state_dict
config
train_losses
```

Checkpoints do not save:

```text
teacher weights
teacher projection
dataset
current batch
hidden states
activations
```

The teacher can be reloaded from the model id if training resumes.

## 12. Visuals And Metrics

At successful epoch end, the monitor saves a loss plot:

```text
loss_curve_epoch_<epoch>.png
```

At the end of training, it also saves:

```text
metrics_detailed.json
training_log.csv
training_summary.log
metrics.json
```

If training crashes before an epoch finishes, plots may not be created.

## 13. Checkpoint Testing

Checkpoint tests live in:

```text
checkpoint_testing/
```

Metadata-only check:

```bash
python3 checkpoint_testing/test_checkpoint.py \
  --checkpoint kd_pipeline/kd_checkpoints/halfway.pt \
  --metadata-only
```

Generation smoke test:

```bash
python3 checkpoint_testing/test_checkpoint.py \
  --checkpoint kd_pipeline/kd_checkpoints/halfway.pt \
  --device cuda \
  --prompt "Question: What is the capital of India?\nAnswer:" \
  --max-new-tokens 40
```

## 14. Full Flow Summary

```text
Dataset JSON
  -> tokenizer
  -> input_ids, attention_mask
  -> GPU

Teacher frozen forward
  -> teacher hidden states
  -> teacher projection
  -> pooled teacher vector p_T

Student trainable forward
  -> student hidden states
  -> student projection
  -> pooled student vector p_S
  -> LM head
  -> logits

Losses
  -> KD MSE loss between p_S and p_T
  -> CE next-token loss from logits
  -> weighted total loss

Backward
  -> update student model
  -> update student projection
  -> teacher stays frozen

Checkpoint
  -> save student
  -> save student projection
  -> save optimizer state
  -> save config and losses
```

