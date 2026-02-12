from __future__ import annotations

import json
from pathlib import Path

from biollm_cls.train import run_training
from tests.unit.test_utils import make_test_config


def test_wake_step_adds_full_batch_to_replay(tmp_path: Path) -> None:
    cfg = make_test_config(output_dir=str(tmp_path), max_steps=1)
    cfg.train.batch_size = 5
    cfg.replay.capacity = 128

    run_training(cfg)

    metrics_path = tmp_path / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "expected at least one logged metrics row"
    row = json.loads(lines[0])

    assert row["replay_added_this_step"] == 5.0
    assert row["replay_size"] == 5.0
