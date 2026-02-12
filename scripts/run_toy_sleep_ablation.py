from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


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


def _format_run_name(prefix: str, mode: str, seed: int) -> str:
    base = f"{mode}-seed{seed}"
    if not prefix:
        return base
    return f"{prefix}-{base}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run toy sleep vs no-sleep ablation batch.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123], help="Seed sweep list")
    parser.add_argument("--device", default="cuda", help="Hydra override for device")
    parser.add_argument("--train-preset", default="rtx4070", help="Hydra train preset override")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "biollm-cls"), help="W&B project name")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"), help="W&B entity slug")
    parser.add_argument("--group", default=None, help="W&B group name")
    parser.add_argument("--run-prefix", default="", help="Prefix to prepend to W&B run names")
    parser.add_argument(
        "--output-root",
        default="outputs/batches",
        help="Directory under repo root for grouped batch outputs",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional train.max_steps override")
    parser.add_argument("--eval-interval", type=int, default=None, help="Optional train.eval_interval override")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional hydra override key=value (repeatable)",
    )
    parser.add_argument("--sleep-only", action="store_true", help="Run only sleep condition")
    parser.add_argument("--nosleep-only", action="store_true", help="Run only no-sleep condition")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue if a run fails")

    args = parser.parse_args()

    if args.sleep_only and args.nosleep_only:
        raise SystemExit("--sleep-only and --nosleep-only are mutually exclusive")

    repo_root = Path(__file__).resolve().parents[1]
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = args.group or f"toy-ablation-{batch_stamp}"

    output_root = repo_root / args.output_root / group
    output_root.mkdir(parents=True, exist_ok=True)

    modes: list[str]
    if args.sleep_only:
        modes = ["sleep"]
    elif args.nosleep_only:
        modes = ["nosleep"]
    else:
        modes = ["sleep", "nosleep"]

    records: list[dict[str, object]] = []
    failures = 0

    for seed in args.seeds:
        for mode in modes:
            is_nosleep = mode == "nosleep"
            run_name = _format_run_name(args.run_prefix, mode, seed)
            run_output_dir = output_root / run_name

            overrides = [
                f"seed={seed}",
                f"device={args.device}",
                f"train={args.train_preset}",
                "benchmark=toy",
                f"train.ablation_no_sleep={'true' if is_nosleep else 'false'}",
                f"logging.wandb_project={args.project}",
                f"logging.wandb_run_name={run_name}",
                f"logging.wandb_group={group}",
                f"logging.output_dir={run_output_dir.as_posix()}",
            ]
            if args.entity:
                overrides.append(f"logging.wandb_entity={args.entity}")
            if args.max_steps is not None:
                overrides.append(f"train.max_steps={args.max_steps}")
            if args.eval_interval is not None:
                overrides.append(f"train.eval_interval={args.eval_interval}")
            overrides.extend(args.extra_override)

            cmd = [sys.executable, "-m", "biollm_cls.cli", "train", *overrides]
            print("\n=== Running:", " ".join(cmd))

            if args.dry_run:
                records.append(
                    {
                        "seed": seed,
                        "mode": mode,
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
                    "mode": mode,
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

    manifest_path = output_root / "batch_manifest.json"
    manifest = {
        "group": group,
        "project": args.project,
        "entity": args.entity,
        "device": args.device,
        "train_preset": args.train_preset,
        "seeds": args.seeds,
        "modes": modes,
        "failures": failures,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nBatch manifest: {manifest_path}")
    print(f"Completed runs: {len(records)}, failures: {failures}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
