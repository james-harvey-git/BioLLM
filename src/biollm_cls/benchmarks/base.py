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


class ContinualBenchmark(Protocol):
    def current_task(self, step: int) -> int: ...

    def sample_batch(self, step: int, batch_size: int | None = None) -> BenchmarkBatch: ...

    def evaluate_old_loss(self, model: NeocortexAdapter, device: torch.device, current_task: int) -> float: ...

    def reward_from_correctness(self, logits: torch.Tensor, targets: torch.Tensor) -> float: ...
