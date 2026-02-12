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

## One-command toy ablation batch (sleep vs no-sleep)

Run the full seed sweep without retyping overrides:

```bash
uv run python scripts/run_toy_sleep_ablation.py \
  --device=cuda \
  --train-preset=rtx4070 \
  --project=biollm-cls \
  --entity=jublett-university-of-oxford
```

Defaults:

- Seeds: `7 42 123`
- Modes: `sleep` and `nosleep`
- Benchmark: `toy`
- Outputs: `outputs/batches/<group>/<run_name>`

Useful options:

```bash
uv run python scripts/run_toy_sleep_ablation.py --max-steps=1500 --eval-interval=10
uv run python scripts/run_toy_sleep_ablation.py --sleep-only
uv run python scripts/run_toy_sleep_ablation.py --nosleep-only
uv run python scripts/run_toy_sleep_ablation.py --dry-run
```

Publish a W&B visual comparison (paired deltas by seed) from existing toy runs:

```bash
uv run python scripts/publish_toy_sleep_comparison_wandb.py \
  --entity=jublett-university-of-oxford \
  --project=biollm-cls
```

The default config now uses `device=auto` and will pick CUDA when available.
For an RTX 4070 (12GB), a good starting run is:

```bash
uv run python -m biollm_cls.cli train \
  device=cuda \
  train=rtx4070 \
  logging.wandb_project=biollm-cls
```

Mac development preset:

```bash
uv run python -m biollm_cls.cli train \
  device=auto \
  train=mac_dev \
  logging.use_wandb=false
```

Manual override equivalent:

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

Qwen 0.5B instruct starting point for a 12GB GPU:

```bash
uv run python scripts/run_qwen_instruction_12gb.py \
  --local-path=/absolute/path/to/instructions.jsonl \
  --entity=jublett-university-of-oxford \
  --project=biollm-cls
```

The launcher defaults to `bf16` and a reduced fast LR for stability.  
Forcing lower precision (`fp16`) or quantized training generally increases instability in this loop.

Equivalent raw override command:

```bash
uv run python -m biollm_cls.cli train \
  device=cuda \
  train=qwen_0_5b_12gb \
  model=hf_causal_lm \
  model.hf_model_name=Qwen/Qwen2.5-0.5B-Instruct \
  model.hf_torch_dtype=fp16 \
  model.vocab_size=16384 \
  benchmark=instruction \
  benchmark.local_path=/absolute/path/to/instructions.jsonl \
  benchmark.prompt_field=instruction \
  benchmark.response_field=output \
  benchmark.task_field=task \
  benchmark.enforce_full_vocab=false \
  replay.batch_size=8 \
  experts.num_experts=8 \
  consolidation.min_sleep_steps=1 \
  consolidation.max_sleep_steps=4
```

## Instruction benchmark (continual, non-toy)

You can switch from the toy stream to an instruction dataset benchmark:

```bash
uv run python -m biollm_cls.cli train \
  model=hf_causal_lm \
  model.hf_model_name=gpt2 \
  model.vocab_size=50257 \
  benchmark=instruction \
  benchmark.dataset_name=tatsu-lab/alpaca \
  benchmark.prompt_field=instruction \
  benchmark.response_field=output \
  benchmark.max_examples=20000 \
  train=rtx4070
```

Using a local dataset instead of Hugging Face datasets:

```bash
uv run python -m biollm_cls.cli train \
  model=hf_causal_lm \
  model.hf_model_name=gpt2 \
  model.vocab_size=50257 \
  benchmark=instruction \
  benchmark.local_path=/absolute/path/to/instructions.jsonl \
  benchmark.prompt_field=instruction \
  benchmark.response_field=output \
  train=rtx4070
```

`instructions.jsonl` should contain one JSON object per line with at least prompt/response fields.

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

## Gaming PC handoff

See `/Users/jamesharvey/Development/Personal-Code-Projects/BioLLM/RUN_ON_4070.md` for full setup and runbook.

Quick CUDA sanity check:

```bash
uv run python scripts/check_cuda.py
```

## Tests

```bash
uv run pytest
```

## Config

Hydra config root is `conf/config.yaml`.
A flattened baseline config snapshot also exists at `mvp.yaml`.
