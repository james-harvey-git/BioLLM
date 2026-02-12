from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from biollm_cls.benchmarks.base import TaskEvalResult


@dataclass
class ContinualMetricsTracker:
    """Tracks task-boundary continual-learning metrics."""

    task_order: list[int] = field(default_factory=list)
    _task_to_col: dict[int, int] = field(default_factory=dict)
    acc_matrix: list[list[float]] = field(default_factory=list)
    loss_matrix: list[list[float]] = field(default_factory=list)
    boundary_steps: list[int] = field(default_factory=list)
    non_finite_step_count: int = 0
    first_non_finite_step: int | None = None

    def observe_step_metrics(self, step: int, metrics: dict[str, float]) -> None:
        if any(not math.isfinite(float(v)) for v in metrics.values()):
            self.non_finite_step_count += 1
            if self.first_non_finite_step is None:
                self.first_non_finite_step = int(step)

    def _ensure_task(self, task_id: int) -> None:
        if task_id in self._task_to_col:
            return
        self._task_to_col[task_id] = len(self.task_order)
        self.task_order.append(task_id)

    def _seen_slice(self, boundary_idx: int) -> tuple[int, list[float], list[float]]:
        n_tasks = len(self.task_order)
        seen = min(boundary_idx + 1, n_tasks)
        row_acc = self.acc_matrix[boundary_idx][:seen]
        row_loss = self.loss_matrix[boundary_idx][:seen]
        return seen, row_acc, row_loss

    def _boundary_metrics(self, boundary_idx: int) -> dict[str, float]:
        seen, row_acc, row_loss = self._seen_slice(boundary_idx)
        seen_acc_avg = float(np.mean(row_acc)) if row_acc else 0.0
        seen_loss_avg = float(np.mean(row_loss)) if row_loss else 0.0

        if seen <= 1:
            forgetting = 0.0
            bwt = 0.0
        else:
            forgetting_terms: list[float] = []
            bwt_terms: list[float] = []
            for j in range(seen - 1):
                prev_values = [self.acc_matrix[k][j] for k in range(boundary_idx) if j < len(self.acc_matrix[k])]
                prev_max = max(prev_values) if prev_values else self.acc_matrix[boundary_idx][j]
                forgetting_terms.append(float(prev_max - self.acc_matrix[boundary_idx][j]))
                diag = self.acc_matrix[j][j] if j < len(self.acc_matrix[j]) else self.acc_matrix[boundary_idx][j]
                bwt_terms.append(float(self.acc_matrix[boundary_idx][j] - diag))
            forgetting = float(np.mean(forgetting_terms))
            bwt = float(np.mean(bwt_terms))

        return {
            "seen_acc_avg": seen_acc_avg,
            "seen_loss_avg": seen_loss_avg,
            "forgetting": forgetting,
            "bwt": bwt,
            "tasks_completed": float(seen),
        }

    def record_boundary(
        self,
        *,
        step: int,
        completed_task_id: int,
        task_results: dict[int, TaskEvalResult],
    ) -> dict[str, float]:
        self._ensure_task(completed_task_id)
        for task_id in sorted(task_results.keys()):
            self._ensure_task(int(task_id))

        row_acc: list[float] = []
        row_loss: list[float] = []
        for task_id in self.task_order:
            result = task_results.get(task_id)
            if result is None or result.n_tokens <= 0:
                row_acc.append(0.0)
                row_loss.append(0.0)
            else:
                row_acc.append(float(result.token_acc))
                row_loss.append(float(result.loss))

        self.acc_matrix.append(row_acc)
        self.loss_matrix.append(row_loss)
        self.boundary_steps.append(int(step))

        boundary_idx = len(self.acc_matrix) - 1
        metrics = self._boundary_metrics(boundary_idx)
        metrics["boundary_index"] = float(boundary_idx)
        metrics["boundary_step"] = float(step)
        return metrics

    def seen_acc_auc(self) -> float:
        if not self.acc_matrix:
            return 0.0

        ys = [self._boundary_metrics(i)["seen_acc_avg"] for i in range(len(self.acc_matrix))]
        if len(ys) == 1:
            return float(ys[0])

        x = np.arange(len(ys), dtype=np.float64)
        auc = np.trapezoid(np.asarray(ys, dtype=np.float64), x)
        return float(auc / (len(ys) - 1))

    def summary(self) -> dict[str, float]:
        if not self.acc_matrix:
            return {
                "final_seen_acc_avg": 0.0,
                "final_forgetting": 0.0,
                "final_bwt": 0.0,
                "seen_acc_auc": 0.0,
                "tasks_completed": 0.0,
                "non_finite_step_count": float(self.non_finite_step_count),
                "first_non_finite_step": -1.0,
            }

        last_idx = len(self.acc_matrix) - 1
        last = self._boundary_metrics(last_idx)
        return {
            "final_seen_acc_avg": float(last["seen_acc_avg"]),
            "final_forgetting": float(last["forgetting"]),
            "final_bwt": float(last["bwt"]),
            "seen_acc_auc": float(self.seen_acc_auc()),
            "tasks_completed": float(last["tasks_completed"]),
            "non_finite_step_count": float(self.non_finite_step_count),
            "first_non_finite_step": float(self.first_non_finite_step if self.first_non_finite_step is not None else -1),
        }
