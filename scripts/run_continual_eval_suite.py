from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


BASELINES: dict[str, list[str]] = {
    "full_cls": [],
    "no_sleep": ["train.ablation_no_sleep=true"],
    "no_ewc": ["ewc.lambda_=0.0"],
    "no_refresh": ["consolidation.refresh.reset_factor=1.0"],
    "no_hippocampus": ["train.ablation_disable_hippocampus=true"],
}


def _run_and_stream(cmd: list[str], cwd: Path) -> tuple[int, float]:
    start = time.time()
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
    return int(proc.returncode), time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Run continual-learning baseline suite on NI8.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-preset", default="ni8_pilot")
    parser.add_argument("--train-preset", default="continual_pilot_4070")
    parser.add_argument("--model-preset", default="hf_causal_lm")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hf-dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "biollm-cls"))
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--group", default=None)
    parser.add_argument("--output-root", default="outputs/eval")
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = args.group or f"cl-ni8-pilot-{stamp}"

    output_root = repo_root / args.output_root / group
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    failures = 0

    for seed in args.seeds:
        for baseline_name, baseline_overrides in BASELINES.items():
            run_name = f"{baseline_name}-seed{seed}"
            run_output = output_root / run_name

            overrides = [
                f"seed={seed}",
                f"device={args.device}",
                f"benchmark={args.benchmark_preset}",
                f"train={args.train_preset}",
                f"model={args.model_preset}",
                f"model.hf_model_name={args.hf_model}",
                f"model.hf_torch_dtype={args.hf_dtype}",
                "benchmark.num_tasks=8",
                "benchmark.switch_every=200",
                "train.max_steps=1600",
                f"logging.wandb_project={args.project}",
                f"logging.wandb_group={group}",
                f"logging.wandb_run_name={run_name}",
                f"logging.output_dir={run_output.as_posix()}",
            ]
            if args.entity:
                overrides.append(f"logging.wandb_entity={args.entity}")

            overrides.extend(baseline_overrides)
            overrides.extend(args.extra_override)

            cmd = [sys.executable, "-m", "biollm_cls.cli", "train", *overrides]
            print("\n=== Running:", " ".join(cmd))

            if args.dry_run:
                records.append(
                    {
                        "seed": seed,
                        "baseline": baseline_name,
                        "run_name": run_name,
                        "command": cmd,
                        "returncode": None,
                        "duration_sec": 0.0,
                    }
                )
                continue

            rc, duration = _run_and_stream(cmd, cwd=repo_root)
            records.append(
                {
                    "seed": seed,
                    "baseline": baseline_name,
                    "run_name": run_name,
                    "command": cmd,
                    "returncode": rc,
                    "duration_sec": round(duration, 3),
                }
            )
            if rc != 0:
                failures += 1
                print(f"Run failed: {run_name} (exit {rc})")
                if not args.continue_on_error:
                    break
        else:
            continue
        break

    manifest = {
        "group": group,
        "project": args.project,
        "entity": args.entity,
        "seeds": args.seeds,
        "baselines": list(BASELINES.keys()),
        "records": records,
        "failures": failures,
    }
    manifest_path = output_root / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Scheduled/finished runs: {len(records)}, failures: {failures}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
