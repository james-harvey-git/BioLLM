from __future__ import annotations

from biollm_cls.metrics_catalog import (
    continual_dashboard_metric_glossary_rows,
    render_metric_glossary_markdown,
    toy_dashboard_metric_glossary_rows,
    training_metric_glossary_rows,
)


def test_training_metric_glossary_contains_core_keys() -> None:
    rows = training_metric_glossary_rows()
    metrics = {r["metric"] for r in rows}
    for required in [
        "task_loss",
        "old_task_loss",
        "replay_added_this_step",
        "pseudo_batch_size",
        "fisher_source_mode",
        "core/*",
        "boundary/*",
        "seen_acc_avg",
        "forgetting",
        "final_seen_acc_avg",
        "seen_acc_auc",
    ]:
        assert required in metrics


def test_markdown_renderer_and_dashboard_glossaries() -> None:
    rows = continual_dashboard_metric_glossary_rows()
    md = render_metric_glossary_markdown("x", rows)
    assert "| Metric | Scope | Direction |" in md
    assert "final_seen_acc_avg" in md

    toy_rows = toy_dashboard_metric_glossary_rows()
    toy_metrics = {r["metric"] for r in toy_rows}
    assert "delta_final_old_task_loss_nosleep_minus_sleep" in toy_metrics
