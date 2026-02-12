from __future__ import annotations

import math

from biollm_cls.benchmarks.base import TaskEvalResult
from biollm_cls.eval.continual_metrics import ContinualMetricsTracker


def test_continual_metrics_formulas() -> None:
    tracker = ContinualMetricsTracker()

    m0 = tracker.record_boundary(
        step=100,
        completed_task_id=0,
        task_results={0: TaskEvalResult(loss=1.0, token_acc=0.80, n_tokens=100)},
    )
    assert math.isclose(m0["seen_acc_avg"], 0.80, rel_tol=1e-6)
    assert math.isclose(m0["forgetting"], 0.0, rel_tol=1e-6)
    assert math.isclose(m0["bwt"], 0.0, rel_tol=1e-6)

    tracker.record_boundary(
        step=200,
        completed_task_id=1,
        task_results={
            0: TaskEvalResult(loss=1.1, token_acc=0.75, n_tokens=100),
            1: TaskEvalResult(loss=0.9, token_acc=0.82, n_tokens=100),
        },
    )

    m2 = tracker.record_boundary(
        step=300,
        completed_task_id=2,
        task_results={
            0: TaskEvalResult(loss=1.2, token_acc=0.70, n_tokens=100),
            1: TaskEvalResult(loss=1.0, token_acc=0.76, n_tokens=100),
            2: TaskEvalResult(loss=0.8, token_acc=0.85, n_tokens=100),
        },
    )

    assert math.isclose(m2["seen_acc_avg"], (0.70 + 0.76 + 0.85) / 3.0, rel_tol=1e-6)
    assert math.isclose(m2["forgetting"], (0.10 + 0.06) / 2.0, rel_tol=1e-6)
    assert math.isclose(m2["bwt"], (-0.10 - 0.06) / 2.0, rel_tol=1e-6)

    summary = tracker.summary()
    assert math.isclose(summary["final_seen_acc_avg"], m2["seen_acc_avg"], rel_tol=1e-6)
    assert math.isclose(summary["final_forgetting"], m2["forgetting"], rel_tol=1e-6)
    assert math.isclose(summary["final_bwt"], m2["bwt"], rel_tol=1e-6)
    assert math.isclose(summary["seen_acc_auc"], 0.785, rel_tol=1e-4)


def test_non_finite_step_tracking() -> None:
    tracker = ContinualMetricsTracker()

    tracker.observe_step_metrics(1, {"task_loss": 1.0, "base_teacher_kl": 0.5})
    tracker.observe_step_metrics(2, {"task_loss": float("nan"), "base_teacher_kl": 0.5})
    tracker.observe_step_metrics(3, {"task_loss": 1.0, "base_teacher_kl": float("inf")})

    assert tracker.non_finite_step_count == 2
    assert tracker.first_non_finite_step == 2
