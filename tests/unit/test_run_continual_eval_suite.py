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


def test_suite_runner_schedules_expected_runs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    module = _load_script_module("run_continual_eval_suite.py")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_continual_eval_suite.py",
            "--dry-run",
            "--seeds",
            "7",
            "42",
            "123",
            "--group",
            "unit-suite",
            "--output-root",
            str(tmp_path),
            "--project",
            "proj",
            "--entity",
            "ent",
        ],
    )

    rc = module.main()
    assert rc == 0

    manifest_path = tmp_path / "unit-suite" / "suite_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["group"] == "unit-suite"
    assert payload["failures"] == 0
    assert len(payload["records"]) == 15

    baselines = {row["baseline"] for row in payload["records"]}
    assert baselines == {"full_cls", "no_sleep", "no_ewc", "no_refresh", "no_hippocampus"}
