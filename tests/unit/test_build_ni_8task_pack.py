from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDatasetInfo:
    builder_name = "fake_builder"


class _FakeDataset(list):
    def __init__(self, rows):
        super().__init__(rows)
        self._fingerprint = "fake-fingerprint"
        self.info = _FakeDatasetInfo()


class _FakeDatasetsModule:
    __version__ = "0.0.test"

    @staticmethod
    def load_dataset(name, config, split):  # type: ignore[no-untyped-def]
        rows = []
        # 10 tasks, each with 800 clean examples.
        for task_id in range(10):
            for i in range(800):
                rows.append(
                    {
                        "task_name": f"task_{task_id:02d}",
                        "inputs": f"instruction task_{task_id} idx_{i}",
                        "targets": [f"output task_{task_id} idx_{i}"],
                    }
                )
        return _FakeDataset(rows)


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                out.append(json.loads(line))
    return out


def test_build_pack_is_deterministic_and_valid(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "datasets", _FakeDatasetsModule())
    module = _load_script_module("build_ni_8task_pack.py")

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    meta_a = module.build_pack(
        output_dir=out_a,
        dataset_name="Muennighoff/natural-instructions",
        dataset_config=None,
        split="train",
        num_tasks=8,
        task_selection_seed=42,
        min_usable_per_task=700,
        train_per_task=500,
        eval_per_task=200,
    )
    meta_b = module.build_pack(
        output_dir=out_b,
        dataset_name="Muennighoff/natural-instructions",
        dataset_config=None,
        split="train",
        num_tasks=8,
        task_selection_seed=42,
        min_usable_per_task=700,
        train_per_task=500,
        eval_per_task=200,
    )

    assert meta_a["selected_tasks"] == meta_b["selected_tasks"]
    assert meta_a["checksums"]["task_list_sha256"] == meta_b["checksums"]["task_list_sha256"]
    assert meta_a["checksums"]["train_jsonl_sha256"] == meta_b["checksums"]["train_jsonl_sha256"]
    assert meta_a["checksums"]["eval_jsonl_sha256"] == meta_b["checksums"]["eval_jsonl_sha256"]

    train_rows = _read_jsonl(out_a / "train.jsonl")
    eval_rows = _read_jsonl(out_a / "eval.jsonl")

    assert len(train_rows) == 8 * 500
    assert len(eval_rows) == 8 * 200

    for row in train_rows + eval_rows:
        assert set(["instruction", "output", "task"]).issubset(set(row.keys()))
        assert row["instruction"].strip()
        assert row["output"].strip()
        assert row["task"].strip()

    overlap = set((r["task"], r["instruction"], r["output"]) for r in train_rows) & set(
        (r["task"], r["instruction"], r["output"]) for r in eval_rows
    )
    assert not overlap
