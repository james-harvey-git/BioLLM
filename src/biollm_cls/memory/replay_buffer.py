from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class Episode:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    reward: float
    teacher_logits: torch.Tensor
    expert_ids: torch.Tensor
    router_probs: torch.Tensor
    step_id: int


class ReservoirReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._memory: list[Episode] = []
        self._seen = 0

    def __len__(self) -> int:
        return len(self._memory)

    def add(self, episode: Episode) -> None:
        self._seen += 1
        if len(self._memory) < self.capacity:
            self._memory.append(episode)
            return

        idx = random.randint(0, self._seen - 1)
        if idx < self.capacity:
            self._memory[idx] = episode

    def sample_uniform(self, batch_size: int) -> list[Episode]:
        if len(self._memory) == 0:
            return []
        n = min(batch_size, len(self._memory))
        return random.sample(self._memory, n)

    def sample_by_expert(self, expert_id: int, n: int) -> list[Episode]:
        candidates = [ep for ep in self._memory if int(expert_id) in ep.expert_ids.view(-1).tolist()]
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": len(self._memory),
            "seen": self._seen,
        }
