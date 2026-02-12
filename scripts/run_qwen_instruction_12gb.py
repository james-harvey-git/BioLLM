from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run_and_stream(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen 0.5B instruct benchmark with 12GB-safe defaults.")
    parser.add_argument("--local-path", required=True, help="Path to local instruction dataset (.jsonl/.json)")
    parser.add_argument("--device", default="cuda", help="Hydra device override")
    parser.add_argument("--train-preset", default="qwen_0_5b_12gb", help="Hydra train preset")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model name")
    parser.add_argument("--model-dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="HF model dtype")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"], help="AMP dtype")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Active vocab cap for runtime")
    parser.add_argument("--prompt-field", default="instruction", help="Prompt field in dataset")
    parser.add_argument("--response-field", default="output", help="Response field in dataset")
    parser.add_argument("--task-field", default="task", help="Task/domain field in dataset")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "biollm-cls"), help="W&B project")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"), help="W&B entity")
    parser.add_argument("--run-name", default=None, help="Optional W&B run name")
    parser.add_argument("--group", default="qwen-instruction", help="Optional W&B group")
    parser.add_argument("--output-dir", default=None, help="Optional local output dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional train.max_steps override")
    parser.add_argument("--eval-interval", type=int, default=None, help="Optional train.eval_interval override")
    parser.add_argument("--replay-batch-size", type=int, default=8, help="Replay batch size")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--fast-lr", type=float, default=3e-4, help="Fast expert/router learning rate")
    parser.add_argument("--reward-alpha", type=float, default=0.2, help="Reward modulation alpha")
    parser.add_argument("--min-sleep-steps", type=int, default=1, help="Minimum sleep steps")
    parser.add_argument("--max-sleep-steps", type=int, default=4, help="Maximum sleep steps")
    parser.add_argument("--enforce-full-vocab", action="store_true", help="Enable strict tokenizer-vocab enforcement")
    parser.add_argument("--hf-trust-remote-code", action="store_true", help="Enable HF trust_remote_code")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override key=value (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command only")

    args = parser.parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists():
        raise SystemExit(f"Dataset path does not exist: {local_path}")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/qwen_instruction/{stamp}"

    overrides = [
        f"device={args.device}",
        f"train={args.train_preset}",
        "model=hf_causal_lm",
        f"model.hf_model_name={args.hf_model}",
        f"model.hf_torch_dtype={args.model_dtype}",
        f"train.amp_dtype={args.amp_dtype}",
        f"model.vocab_size={args.vocab_size}",
        "benchmark=instruction",
        f"benchmark.local_path={local_path.resolve().as_posix()}",
        f"benchmark.prompt_field={args.prompt_field}",
        f"benchmark.response_field={args.response_field}",
        f"benchmark.task_field={args.task_field}",
        f"benchmark.enforce_full_vocab={'true' if args.enforce_full_vocab else 'false'}",
        f"replay.batch_size={args.replay_batch_size}",
        f"experts.num_experts={args.num_experts}",
        f"experts.fast_lr={args.fast_lr}",
        f"experts.reward_alpha={args.reward_alpha}",
        f"consolidation.min_sleep_steps={args.min_sleep_steps}",
        f"consolidation.max_sleep_steps={args.max_sleep_steps}",
        f"logging.wandb_project={args.project}",
        f"logging.wandb_group={args.group}",
        f"logging.output_dir={output_dir}",
        f"model.hf_trust_remote_code={'true' if args.hf_trust_remote_code else 'false'}",
    ]

    if args.entity:
        overrides.append(f"logging.wandb_entity={args.entity}")
    if args.run_name:
        overrides.append(f"logging.wandb_run_name={args.run_name}")
    if args.max_steps is not None:
        overrides.append(f"train.max_steps={args.max_steps}")
    if args.eval_interval is not None:
        overrides.append(f"train.eval_interval={args.eval_interval}")
    overrides.extend(args.extra_override)

    cmd = [sys.executable, "-m", "biollm_cls.cli", "train", *overrides]
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    return _run_and_stream(cmd, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
