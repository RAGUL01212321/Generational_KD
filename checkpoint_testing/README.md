# Checkpoint Testing

Standalone tools for validating KD checkpoints without touching the training
pipeline.

## Quick Checks

Inspect checkpoint metadata without loading the Apollo model:

```bash
python3 checkpoint_testing/test_checkpoint.py \
  --checkpoint kd_pipeline/kd_checkpoints/step_5001.pt \
  --metadata-only
```

Load the student weights and run a tiny generation smoke test:

```bash
python3 checkpoint_testing/test_checkpoint.py \
  --checkpoint kd_pipeline/kd_checkpoints/step_5001.pt \
  --device cuda \
  --prompt "What is diabetes?" \
  --max-new-tokens 40
```

For low VRAM machines, use CPU for validation:

```bash
python3 checkpoint_testing/test_checkpoint.py \
  --checkpoint kd_pipeline/kd_checkpoints/step_5001.pt \
  --device cpu \
  --load-model-only
```

## What It Checks

- The checkpoint can be opened by `torch.load`.
- Required keys exist:
  - `student_state_dict`
  - `proj_student_state_dict`
  - `config`
  - `global_step`
- The student model can load `student_state_dict`.
- The projection layer can load `proj_student_state_dict`.
- Optional text generation works from the loaded student model.

The teacher model is not loaded here.
