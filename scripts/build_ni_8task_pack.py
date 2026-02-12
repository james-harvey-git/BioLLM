from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    task: str
    instruction: str
    output: str


def _normalize_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _first_non_empty_target(raw: Any) -> str:
    if isinstance(raw, str):
        return _normalize_text(raw)
    if isinstance(raw, list):
        for item in raw:
            text = _normalize_text(str(item))
            if text:
                return text
    return ""


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_task_shuffle(rows: list[Example], base_seed: int, task_name: str) -> list[Example]:
    task_seed = int(hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(base_seed ^ task_seed)
    out = list(rows)
    rng.shuffle(out)
    return out


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _row_iterator(ds, *, verbose: bool, progress_every: int):
    total = len(ds) if hasattr(ds, "__len__") else None
    try:
        from tqdm.auto import tqdm  # type: ignore

        yield from tqdm(ds, total=total, desc="Cleaning rows", unit="row")
        return
    except Exception:
        pass

    for idx, row in enumerate(ds, start=1):
        if verbose and progress_every > 0 and idx % progress_every == 0:
            if total is not None and total > 0:
                pct = 100.0 * idx / total
                _log(f"[build_ni_8task_pack] processed {idx}/{total} rows ({pct:.1f}%)", verbose=True)
            else:
                _log(f"[build_ni_8task_pack] processed {idx} rows", verbose=True)
        yield row


def _load_dataset_split(
    load_dataset_fn,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    verification_mode: str,
):
    def _call(mode: str):
        if mode == "strict":
            return load_dataset_fn(dataset_name, dataset_config, split=split)
        if mode == "no_checks":
            try:
                return load_dataset_fn(dataset_name, dataset_config, split=split, verification_mode="no_checks")
            except TypeError:
                # Backward compatibility for older datasets versions.
                return load_dataset_fn(dataset_name, dataset_config, split=split, ignore_verifications=True)
        raise ValueError(f"Unsupported verification_mode: {mode}")

    if verification_mode == "strict":
        return _call("strict"), "strict"
    if verification_mode == "no_checks":
        return _call("no_checks"), "no_checks"

    # auto mode: strict first, then fallback only for split-verification mismatch.
    try:
        return _call("strict"), "strict"
    except Exception as exc:
        exc_text = f"{type(exc).__name__}: {exc}"
        if "ExpectedMoreSplitsError" not in exc_text:
            raise
        return _call("no_checks"), "no_checks"


def build_pack(
    *,
    output_dir: Path,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    num_tasks: int,
    task_selection_seed: int,
    min_usable_per_task: int,
    train_per_task: int,
    eval_per_task: int,
    verification_mode: str = "auto",
    verbose: bool = True,
    progress_every: int = 50_000,
) -> dict[str, Any]:
    try:
        import datasets
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required. Install deps with `uv sync`.") from exc

    ds, verification_mode_used = _load_dataset_split(
        load_dataset,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        verification_mode=verification_mode,
    )
    _log(
        (
            "[build_ni_8task_pack] loaded dataset "
            f"{dataset_name} split={split} rows={len(ds)} verification_mode={verification_mode_used}"
        ),
        verbose=verbose,
    )
    grouped: dict[str, list[Example]] = defaultdict(list)

    for row in _row_iterator(ds, verbose=verbose, progress_every=progress_every):
        task = _normalize_text(str(row.get("task_name", "")))
        instruction = _normalize_text(str(row.get("inputs", "")))
        output = _first_non_empty_target(row.get("targets"))
        if not task or not instruction or not output:
            continue
        grouped[task].append(Example(task=task, instruction=instruction, output=output))
    _log(
        f"[build_ni_8task_pack] cleaned rows into {len(grouped)} tasks (pre-filter)",
        verbose=verbose,
    )

    candidates = [task for task, rows in grouped.items() if len(rows) >= min_usable_per_task]
    if len(candidates) < num_tasks:
        raise ValueError(
            f"Only {len(candidates)} tasks have >= {min_usable_per_task} usable rows; need {num_tasks}."
        )
    _log(
        f"[build_ni_8task_pack] {len(candidates)} tasks passed min_usable_per_task={min_usable_per_task}",
        verbose=verbose,
    )

    candidates = sorted(candidates)
    rng = random.Random(task_selection_seed)
    rng.shuffle(candidates)
    selected_tasks = candidates[:num_tasks]
    _log(f"[build_ni_8task_pack] selected {num_tasks} tasks with seed={task_selection_seed}", verbose=verbose)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    per_task_counts: dict[str, dict[str, int]] = {}

    task_iterable = selected_tasks
    try:
        from tqdm.auto import tqdm  # type: ignore

        task_iterable = tqdm(selected_tasks, desc="Building task splits", unit="task")
    except Exception:
        pass

    for task in task_iterable:
        task_rows = _stable_task_shuffle(grouped[task], task_selection_seed, task)
        required = train_per_task + eval_per_task
        if len(task_rows) < required:
            raise ValueError(
                f"Task '{task}' has {len(task_rows)} usable rows after cleaning; requires {required}."
            )

        eval_slice = task_rows[:eval_per_task]
        train_slice = task_rows[eval_per_task : eval_per_task + train_per_task]

        train_rows.extend(
            {
                "task": task,
                "instruction": ex.instruction,
                "output": ex.output,
            }
            for ex in train_slice
        )
        eval_rows.extend(
            {
                "task": task,
                "instruction": ex.instruction,
                "output": ex.output,
            }
            for ex in eval_slice
        )

        per_task_counts[task] = {
            "usable": len(task_rows),
            "train": len(train_slice),
            "eval": len(eval_slice),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    metadata_path = output_dir / "metadata.json"

    _jsonl_write(train_path, train_rows)
    _jsonl_write(eval_path, eval_rows)

    dataset_info = getattr(ds, "info", None)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "num_tasks": num_tasks,
        "task_selection_seed": task_selection_seed,
        "min_usable_per_task": min_usable_per_task,
        "train_per_task": train_per_task,
        "eval_per_task": eval_per_task,
        "selected_tasks": selected_tasks,
        "counts": per_task_counts,
        "rows": {
            "train": len(train_rows),
            "eval": len(eval_rows),
        },
        "source": {
            "datasets_version": datasets.__version__,
            "dataset_fingerprint": getattr(ds, "_fingerprint", None),
            "builder_name": getattr(dataset_info, "builder_name", None),
            "dataset_size": len(ds),
            "verification_mode_used": verification_mode_used,
        },
        "checksums": {
            "train_jsonl_sha256": _sha256_file(train_path),
            "eval_jsonl_sha256": _sha256_file(eval_path),
            "task_list_sha256": hashlib.sha256("\n".join(selected_tasks).encode("utf-8")).hexdigest(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(
        (
            "[build_ni_8task_pack] wrote outputs "
            f"train={train_path} eval={eval_path} metadata={metadata_path}"
        ),
        verbose=verbose,
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic NI 8-task continual-learning pack.")
    parser.add_argument("--output-dir", default="data/ni8", help="Output directory for train/eval/metadata files")
    parser.add_argument("--dataset-name", default="Muennighoff/natural-instructions")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--task-selection-seed", type=int, default=42)
    parser.add_argument("--min-usable-per-task", type=int, default=700)
    parser.add_argument("--train-per-task", type=int, default=500)
    parser.add_argument("--eval-per-task", type=int, default=200)
    parser.add_argument(
        "--verification-mode",
        choices=["auto", "strict", "no_checks"],
        default="auto",
        help="Dataset split verification mode. 'auto' retries with no_checks on split-verification mismatch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Fallback progress print interval when tqdm is unavailable.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress/log output.")
    args = parser.parse_args()

    metadata = build_pack(
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        num_tasks=args.num_tasks,
        task_selection_seed=args.task_selection_seed,
        min_usable_per_task=args.min_usable_per_task,
        train_per_task=args.train_per_task,
        eval_per_task=args.eval_per_task,
        verification_mode=args.verification_mode,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
