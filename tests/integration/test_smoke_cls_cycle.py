from __future__ import annotations

from pathlib import Path

from biollm_cls.train import run_training
from tests.unit.test_utils import make_test_config


def test_smoke_cls_cycle_and_ablation(tmp_path: Path) -> None:
    with_sleep_cfg = make_test_config(output_dir=str(tmp_path / "with_sleep"), ablation_no_sleep=False, max_steps=260)
    with_sleep_cfg.consolidation.refresh.kl_threshold = 1.0
    with_sleep = run_training(with_sleep_cfg)

    no_sleep_cfg = make_test_config(output_dir=str(tmp_path / "no_sleep"), ablation_no_sleep=True, max_steps=260)
    no_sleep = run_training(no_sleep_cfg)

    assert with_sleep["sleep_count"] >= 1
    assert with_sleep["base_updates_in_sleep"] > 0
    assert with_sleep["refresh_count_total"] >= 1

    # Allow small variance but require sleep-enabled forgetting to not be worse.
    assert with_sleep["final_forgetting_index"] <= no_sleep["final_forgetting_index"] + 0.15
