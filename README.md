# BioLLM CLS MVP

A research MVP implementing a complementary learning systems (CLS) architecture for continual learning:

- Wake phase: fast sparse hippocampal adaptation (MoE experts + router)
- Sleep phase: slow neocortical consolidation (distillation + EWC)
- Episodic replay buffer with shuffled replay
- Expert refresh/recycling after successful consolidation
- Sleep-pressure scheduler using saturation, novelty, and forgetting

## Reproducible setup (`uv`)

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --frozen
```

## Train

```bash
uv run python -m biollm_cls.cli train
```

The default config now uses `device=auto` and will pick CUDA when available.
For an RTX 4070 (12GB), a good starting run is:

```bash
uv run python -m biollm_cls.cli train \
  train.batch_size=24 \
  replay.batch_size=128 \
  train.amp_enabled=true \
  train.amp_dtype=fp16 \
  train.allow_tf32=true
```

Hydra overrides are supported:

```bash
uv run python -m biollm_cls.cli train experts.num_experts=64 train.max_steps=2000
```

## Hugging Face models

The neocortex can be switched from the internal tiny model to a Hugging Face causal LM:

```bash
uv run python -m biollm_cls.cli train \
  model=hf_causal_lm \
  model.hf_model_name=gpt2 \
  model.vocab_size=1024 \
  train.batch_size=8
```

Notes:

- `model.provider=tiny|hf` controls which neocortex backend is used.
- For `hf`, this MVP injects the hippocampal delta at the final hidden state before the LM head.
- `model.vocab_size` acts as an active-vocab cap for memory/compute during toy training.

## Weights & Biases (online/offline, artifacts, checkpoints)

The trainer always writes local logs to `metrics.jsonl`.  
W&B can run in `offline` or `online` mode via config/env:

- `logging.wandb_mode=offline|online`
- `WANDB_MODE=offline|online`

Set up online W&B securely:

```bash
export WANDB_API_KEY="your_key_here"
export WANDB_ENTITY="your_team_or_user"
export WANDB_MODE="online"
```

Then run:

```bash
uv run python -m biollm_cls.cli train logging.wandb_project=biollm-cls
```

Optional run metadata overrides:

```bash
uv run python -m biollm_cls.cli train \
  logging.wandb_run_name=cls-sleep-exp1 \
  logging.wandb_group=ablation-a \
  logging.wandb_tags=[cls,sleep,ewc] \
  logging.wandb_resume=allow
```

What is logged to W&B:

- Per-step scalar metrics (task loss, forgetting index, sleep metrics, KL, etc.)
- Final summary metrics
- `validated_config.json` artifact
- `run_metadata.json` artifact
- Periodic and final model checkpoints as model artifacts

If `WANDB_API_KEY` is not set and mode is online, the run automatically falls back to offline mode.

## Tests

```bash
uv run pytest
```

## Config

Hydra config root is `conf/config.yaml`.
A flattened baseline config snapshot also exists at `mvp.yaml`.
