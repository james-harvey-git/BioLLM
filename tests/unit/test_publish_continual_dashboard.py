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


class _FakeRun:
    def __init__(self, run_id: str, name: str, group: str, summary: dict, history: list[dict]):
        self.id = run_id
        self.name = name
        self.group = group
        self.summary = summary
        self._history = history

    def scan_history(self):
        return list(self._history)


class _FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, path, filters=None):  # type: ignore[no-untyped-def]
        if filters is None:
            return list(self._runs)
        group = filters.get("group")
        return [r for r in self._runs if r.group == group]


class _FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *args):  # type: ignore[no-untyped-def]
        self.rows.append(args)


class _FakePlot:
    @staticmethod
    def bar(table, x, y, title=None):  # type: ignore[no-untyped-def]
        return {"kind": "bar", "x": x, "y": y, "title": title, "rows": len(table.rows)}

    @staticmethod
    def line(table, x, y, title=None):  # type: ignore[no-untyped-def]
        return {"kind": "line", "x": x, "y": y, "title": title, "rows": len(table.rows)}


class _FakeAnalysisRun:
    def __init__(self):
        self.summary = {}
        self.logged = []

    def log(self, payload):  # type: ignore[no-untyped-def]
        self.logged.append(payload)

    def finish(self):
        return None


class _FakeWandb:
    def __init__(self, runs):
        self._runs = runs
        self.plot = _FakePlot()

    def Api(self):  # noqa: N802
        return _FakeApi(self._runs)

    def init(self, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeAnalysisRun()

    def Table(self, columns):  # noqa: N802
        return _FakeTable(columns)


def test_publish_continual_dashboard_writes_csvs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    module = _load_script_module("publish_continual_dashboard_wandb.py")

    fake_runs = []
    for baseline in ("full_cls", "no_sleep"):
        for seed in (7, 42):
            run_name = f"{baseline}-seed{seed}"
            summary = {
                "final_seen_acc_avg": 0.6 if baseline == "full_cls" else 0.5,
                "final_forgetting": 0.1 if baseline == "full_cls" else 0.2,
                "final_bwt": -0.05,
                "seen_acc_auc": 0.55 if baseline == "full_cls" else 0.45,
            }
            history = [
                {
                    "step": 100,
                    "boundary_index": 0,
                    "boundary_step": 100,
                    "seen_acc_avg": 0.5,
                    "seen_loss_avg": 1.0,
                    "forgetting": 0.0,
                    "bwt": 0.0,
                    "tasks_completed": 1,
                    "task_acc/0": 0.5,
                    "task_loss/0": 1.0,
                    "task_tokens/0": 100,
                },
                {
                    "step": 200,
                    "boundary_index": 1,
                    "boundary_step": 200,
                    "seen_acc_avg": 0.55,
                    "seen_loss_avg": 0.9,
                    "forgetting": 0.1,
                    "bwt": -0.1,
                    "tasks_completed": 2,
                    "task_acc/0": 0.45,
                    "task_loss/0": 1.1,
                    "task_tokens/0": 100,
                    "task_acc/1": 0.65,
                    "task_loss/1": 0.8,
                    "task_tokens/1": 100,
                },
            ]
            fake_runs.append(_FakeRun(run_id=f"{baseline}-{seed}", name=run_name, group="g1", summary=summary, history=history))

    monkeypatch.setattr(module, "wandb", _FakeWandb(fake_runs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_continual_dashboard_wandb.py",
            "--entity",
            "ent",
            "--project",
            "proj",
            "--group",
            "g1",
            "--output-root",
            str(tmp_path),
        ],
    )

    rc = module.main()
    assert rc == 0

    out_dir = tmp_path / "g1"
    assert (out_dir / "suite_summary.csv").exists()
    assert (out_dir / "boundary_metrics.csv").exists()
    assert (out_dir / "task_matrix.csv").exists()

    # Basic shape sanity checks.
    suite_lines = (out_dir / "suite_summary.csv").read_text(encoding="utf-8").strip().splitlines()
    boundary_lines = (out_dir / "boundary_metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    task_lines = (out_dir / "task_matrix.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(suite_lines) > 1
    assert len(boundary_lines) > 1
    assert len(task_lines) > 1
