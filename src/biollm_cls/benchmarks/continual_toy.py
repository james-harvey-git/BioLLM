from __future__ import annotations

import torch
import torch.nn.functional as F

from biollm_cls.benchmarks.base import BenchmarkBatch
from biollm_cls.models.base import NeocortexAdapter


class ContinualToyBenchmark:
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        switch_every: int = 200,
        num_tasks: int = 4,
        heldout_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.switch_every = switch_every
        self.num_tasks = num_tasks
        self._gen = torch.Generator().manual_seed(seed)
        self.heldout: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for task_id in range(num_tasks):
            inputs = torch.randint(0, vocab_size, (heldout_size, seq_len), generator=self._gen)
            targets = self._transform(task_id, inputs)
            self.heldout[task_id] = (inputs, targets)

    def current_task(self, step: int) -> int:
        return (step // self.switch_every) % self.num_tasks

    def _transform(self, task_id: int, inputs: torch.Tensor) -> torch.Tensor:
        if task_id == 0:
            return (inputs + 1) % self.vocab_size
        if task_id == 1:
            return (inputs * 2 + 3) % self.vocab_size
        if task_id == 2:
            return torch.bitwise_xor(inputs, torch.full_like(inputs, 0x1F)) % self.vocab_size
        return (inputs + (inputs % 7)) % self.vocab_size

    def sample_batch(self, step: int, batch_size: int | None = None) -> BenchmarkBatch:
        bsz = batch_size or self.batch_size
        task_id = self.current_task(step)
        inputs = torch.randint(0, self.vocab_size, (bsz, self.seq_len), generator=self._gen)
        targets = self._transform(task_id, inputs)
        return BenchmarkBatch(input_ids=inputs, target_ids=targets, task_id=task_id)

    def evaluate_old_loss(self, model: NeocortexAdapter, device: torch.device, current_task: int) -> float:
        old_tasks = [tid for tid in range(self.num_tasks) if tid != current_task]
        if not old_tasks:
            return 0.0

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for task_id in old_tasks:
                inputs, targets = self.heldout[task_id]
                logits = model.forward_base(inputs.to(device))
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.to(device).view(-1),
                    ignore_index=-100,
                )
                losses.append(float(loss.item()))
        model.train()
        return float(sum(losses) / len(losses))

    @staticmethod
    def reward_from_correctness(logits: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = targets != -100
            if not mask.any():
                return 0.0
            acc = (pred[mask] == targets[mask]).float().mean().item()
        return 2.0 * acc - 1.0
