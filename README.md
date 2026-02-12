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

## Continual Learning Evaluation v1 (NI8 pilot)

Build the deterministic 8-task Natural Instructions pack:

```bash
uv run python scripts/build_ni_8task_pack.py
```

This writes:

- `data/ni8/train.jsonl`
- `data/ni8/eval.jsonl`
- `data/ni8/metadata.json`

Run the 5-baseline continual suite (`full_cls`, `no_sleep`, `no_ewc`, `no_refresh`, `no_hippocampus`) across 3 seeds:

```bash
uv run python scripts/run_continual_eval_suite.py \
  --entity=jublett-university-of-oxford \
  --project=biollm-cls
```

Default suite settings:

- Benchmark preset: `benchmark=ni8_pilot`
- Train preset: `train=continual_pilot_4070`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (`bf16`)
- Seeds: `7 42 123`
- Group: `cl-ni8-pilot-<timestamp>`

Publish W&B dashboard visuals + local CSV reports for a completed suite group:

```bash
uv run python scripts/publish_continual_dashboard_wandb.py \
  --entity=jublett-university-of-oxford \
  --project=biollm-cls \
  --group=cl-ni8-pilot-YYYYMMDD_HHMMSS
```

Local report outputs:

- `outputs/eval/<group>/suite_summary.csv`
- `outputs/eval/<group>/boundary_metrics.csv`
- `outputs/eval/<group>/task_matrix.csv`

### Continual Metrics Definitions (Task-Boundary Metrics)

Implementation reference:
- `src/biollm_cls/eval/continual_metrics.py`

At each task boundary (task switch, plus final step), we evaluate all seen tasks and update two matrices:

- `A[i][j]`: token accuracy on task `j` measured after completing boundary `i`
- `L[i][j]`: token loss on task `j` measured after completing boundary `i`

Conventions:

- `i` is boundary index in task order (0-based)
- `j` is task index among seen tasks
- first boundary (`i=0`) has no prior tasks, so forgetting/BWT are defined as `0`

Core metrics (computed exactly):

- `seen_acc_avg_i = mean(A[i][j] for j <= i)`
  - Average token accuracy over all tasks seen so far at boundary `i`.
  - Higher is better.

- `seen_loss_avg_i = mean(L[i][j] for j <= i)`
  - Average token loss over all tasks seen so far at boundary `i`.
  - Lower is better.

- `forgetting_i = mean(max_{k <= i-1} A[k][j] - A[i][j] for j < i)`
  - For each prior task `j`, measure drop from its best past accuracy to current accuracy, then average.
  - Higher means more catastrophic forgetting. Lower is better.

- `bwt_i = mean(A[i][j] - A[j][j] for j < i)`
  - Backward transfer relative to each task’s own post-training boundary (`A[j][j]`).
  - Positive: old tasks improved after later learning.
  - Negative: old tasks degraded after later learning.

Final summary metrics:

- `final_seen_acc_avg`: `seen_acc_avg_i` at the last boundary
- `final_forgetting`: `forgetting_i` at the last boundary
- `final_bwt`: `bwt_i` at the last boundary

Area-under-trajectory metric:

- `seen_acc_auc`
  - Trapezoidal AUC over boundary trajectory of `seen_acc_avg`, normalized by number of boundary intervals.
  - Interpreted as average retention quality across the full continual stream (not just final checkpoint).
  - Higher is better.

Numerical health metrics:

- `non_finite_step_count`
  - Count of training steps where tracked scalar metrics contained non-finite values (`NaN`/`Inf`).

- `first_non_finite_step`
  - First step index with non-finite metrics, or `-1` if none observed.

Batch supervision coverage metrics:

- `batch_supervised_tokens`
  - Number of label positions in the current batch that are trainable (`target != -100`).

- `batch_supervised_fraction`
  - Ratio `batch_supervised_tokens / total_tokens_in_batch`.
  - If this is near `0`, the wake loss signal is weak/degenerate.

Per-task eval coverage (important for interpretation):

- Coverage means how many valid supervised eval tokens each task contributes at boundary evaluation.
- Tracked as `task_tokens/{task_id}` in boundary logs.
- If a task has `task_tokens/{task_id} = 0`, then `task_acc/{task_id}` and `task_loss/{task_id}` are not meaningful for that task at that boundary.
- Continual metrics are most reliable when all tasks have consistently non-zero eval token coverage.

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
- Metric glossary table (`metric_glossary/table`) with definitions for all logged metrics
- `validated_config.json` artifact
- `run_metadata.json` artifact
- Periodic and final model checkpoints as model artifacts
- Run artifacts: `metric_glossary.json` and `metric_glossary.md`

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
