# Run on RTX 4070 (Windows/Linux gaming PC)

This project is developed on macOS but can be trained on a CUDA workstation.

## 1) Clone and setup

```bash
git clone <your_repo_url>
cd BioLLM
uv python install 3.12
uv venv --python 3.12
uv sync --frozen
```

## 2) Verify CUDA is visible to PyTorch

```bash
uv run python scripts/check_cuda.py
```

Expected: `"cuda_available": true` and device name contains RTX 4070.

## 3) Set W&B environment

### Bash / WSL

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='jubilett-university-of-oxford-org'
export WANDB_MODE='online'
```

### PowerShell

```powershell
$env:WANDB_API_KEY = '...'
$env:WANDB_ENTITY = 'jubilett-university-of-oxford-org'
$env:WANDB_MODE = 'online'
```

## 4) Run baseline CLS training on 4070 preset

```bash
uv run python -m biollm_cls.cli train \
  device=cuda \
  train=rtx4070 \
  logging.wandb_project=biollm-cls
```

## 5) Optional Hugging Face neocortex run

```bash
uv run python -m biollm_cls.cli train \
  device=cuda \
  model=hf_causal_lm \
  model.hf_model_name=gpt2 \
  model.vocab_size=1024 \
  train=rtx4070 \
  train.batch_size=8 \
  replay.batch_size=32 \
  logging.wandb_project=biollm-cls
```

## 6) If you hit out-of-memory (OOM)

Tune down in this order:

1. `train.batch_size`
2. `replay.batch_size`
3. `model.vocab_size` (HF mode)
4. turn off compile: `train.compile_model=false`

Example conservative override:

```bash
uv run python -m biollm_cls.cli train \
  device=cuda \
  train=rtx4070 \
  train.batch_size=12 \
  replay.batch_size=64
```

## 7) Artifacts

- Local metrics: `outputs/<run>/metrics.jsonl`
- Checkpoints: `outputs/<run>/checkpoints/`
- W&B artifacts/runs: your configured project/entity
