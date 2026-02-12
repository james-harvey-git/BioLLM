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
export WANDB_ENTITY='jublett-university-of-oxford'
export WANDB_MODE='online'
```

### PowerShell

```powershell
$env:WANDB_API_KEY = '...'
$env:WANDB_ENTITY = 'jublett-university-of-oxford'
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

## 5b) Instruction benchmark run (recommended for real evaluation)

```bash
uv run python -m biollm_cls.cli train \
  device=cuda \
  model=hf_causal_lm \
  model.hf_model_name=gpt2 \
  model.vocab_size=50257 \
  benchmark=instruction \
  benchmark.dataset_name=tatsu-lab/alpaca \
  benchmark.prompt_field=instruction \
  benchmark.response_field=output \
  benchmark.max_examples=20000 \
  train=rtx4070 \
  logging.wandb_project=biollm-cls
```

For private/local data, use `benchmark.local_path=/absolute/path/to/file.jsonl`.

## 5c) Qwen 0.5B instruct preset for 12GB VRAM

Simple launcher:

```bash
uv run python scripts/run_qwen_instruction_12gb.py \
  --local-path=/absolute/path/to/instructions.jsonl \
  --entity=jublett-university-of-oxford \
  --project=biollm-cls
```

This launcher uses `bf16` + softer fast updates by default for stability on 12GB cards.

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
  consolidation.max_sleep_steps=4 \
  logging.wandb_project=biollm-cls \
  logging.wandb_entity=jublett-university-of-oxford
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

## 8) One-command toy sleep/no-sleep sweep

```bash
uv run python scripts/run_toy_sleep_ablation.py \
  --device=cuda \
  --train-preset=rtx4070 \
  --project=biollm-cls \
  --entity=jublett-university-of-oxford
```

This runs seeds `7/42/123` for both conditions and writes a manifest to:

- `outputs/batches/<group>/batch_manifest.json`

To publish a W&B visual comparison after runs complete:

```bash
uv run python scripts/publish_toy_sleep_comparison_wandb.py \
  --entity=jublett-university-of-oxford \
  --project=biollm-cls
```
