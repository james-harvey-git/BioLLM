from __future__ import annotations

import json
from pathlib import Path

from biollm_cls.train import run_training
from tests.unit.test_utils import make_test_config


def _load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_smoke_cls_cycle_and_ablation(tmp_path: Path) -> None:
    with_sleep_cfg = make_test_config(output_dir=str(tmp_path / "with_sleep"), ablation_no_sleep=False, max_steps=220)
    with_sleep_cfg.consolidation.refresh.kl_threshold = 1.0
    with_sleep = run_training(with_sleep_cfg)

    no_sleep_cfg = make_test_config(output_dir=str(tmp_path / "no_sleep"), ablation_no_sleep=True, max_steps=220)
    no_sleep = run_training(no_sleep_cfg)

    assert with_sleep["sleep_count"] >= 1
    assert with_sleep["base_updates_in_sleep"] > 0
    assert with_sleep["refresh_count_total"] >= 1
    assert no_sleep["sleep_count"] == 0
    assert with_sleep["tasks_completed"] >= 1
    assert "final_seen_acc_avg" in with_sleep
    assert "final_forgetting" in with_sleep
    assert "final_bwt" in with_sleep
    assert "seen_acc_auc" in with_sleep

    # Allow small variance but require sleep-enabled forgetting to not be worse.
    assert with_sleep["final_forgetting_index"] <= no_sleep["final_forgetting_index"] + 0.15


def test_ablation_toggles_behave_as_expected(tmp_path: Path) -> None:
    no_refresh_cfg = make_test_config(
        output_dir=str(tmp_path / "no_refresh"),
        ablation_no_sleep=False,
        max_steps=220,
    )
    no_refresh_cfg.consolidation.refresh.reset_factor = 1.0
    no_refresh_cfg.consolidation.refresh.kl_threshold = 1.0
    no_refresh = run_training(no_refresh_cfg)
    assert no_refresh["refresh_count_total"] == 0.0

    no_ewc_cfg = make_test_config(
        output_dir=str(tmp_path / "no_ewc"),
        ablation_no_sleep=False,
        max_steps=220,
    )
    no_ewc_cfg.ewc.lambda_ = 0.0
    run_training(no_ewc_cfg)
    no_ewc_metrics = _load_metrics(tmp_path / "no_ewc" / "metrics.jsonl")
    assert any(abs(float(row.get("ewc_penalty", 0.0))) < 1e-12 for row in no_ewc_metrics)

    no_hip_cfg = make_test_config(
        output_dir=str(tmp_path / "no_hip"),
        ablation_no_sleep=False,
        max_steps=220,
        ablation_disable_hippocampus=True,
    )
    run_training(no_hip_cfg)
    no_hip_metrics = _load_metrics(tmp_path / "no_hip" / "metrics.jsonl")
    step_rows = [row for row in no_hip_metrics if "fast_update_applied" in row]
    assert step_rows
    assert all(float(row.get("fast_update_applied", 1.0)) == 0.0 for row in step_rows)
    assert all(abs(float(row.get("expert_util_entropy", 0.0))) < 1e-12 for row in step_rows)
