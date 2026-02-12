from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from biollm_cls.models.base import NeocortexAdapter


@dataclass
class BenchmarkBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    task_id: int


@dataclass
class TaskEvalResult:
    loss: float
    token_acc: float
    n_tokens: int


class ContinualBenchmark(Protocol):
    def current_task(self, step: int) -> int: ...

    def sample_batch(self, step: int, batch_size: int | None = None) -> BenchmarkBatch: ...

    def evaluate_task_set(
        self,
        model: NeocortexAdapter,
        device: torch.device,
        task_ids: list[int],
    ) -> dict[int, TaskEvalResult]: ...

    def evaluate_old_loss(self, model: NeocortexAdapter, device: torch.device, current_task: int) -> float: ...

    def reward_from_correctness(self, logits: torch.Tensor, targets: torch.Tensor) -> float: ...
