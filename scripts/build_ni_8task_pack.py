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
) -> dict[str, Any]:
    try:
        import datasets
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required. Install deps with `uv sync`.") from exc

    ds = load_dataset(dataset_name, dataset_config, split=split)
    grouped: dict[str, list[Example]] = defaultdict(list)

    for row in ds:
        task = _normalize_text(str(row.get("task_name", "")))
        instruction = _normalize_text(str(row.get("inputs", "")))
        output = _first_non_empty_target(row.get("targets"))
        if not task or not instruction or not output:
            continue
        grouped[task].append(Example(task=task, instruction=instruction, output=output))

    candidates = [task for task, rows in grouped.items() if len(rows) >= min_usable_per_task]
    if len(candidates) < num_tasks:
        raise ValueError(
            f"Only {len(candidates)} tasks have >= {min_usable_per_task} usable rows; need {num_tasks}."
        )

    candidates = sorted(candidates)
    rng = random.Random(task_selection_seed)
    rng.shuffle(candidates)
    selected_tasks = candidates[:num_tasks]

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    per_task_counts: dict[str, dict[str, int]] = {}

    for task in selected_tasks:
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
        },
        "checksums": {
            "train_jsonl_sha256": _sha256_file(train_path),
            "eval_jsonl_sha256": _sha256_file(eval_path),
            "task_list_sha256": hashlib.sha256("\n".join(selected_tasks).encode("utf-8")).hexdigest(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
    )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
