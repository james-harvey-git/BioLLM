from __future__ import annotations

import random

import torch

from biollm_cls.memory.replay_buffer import Episode, ReservoirReplayBuffer


def _episode(step_id: int) -> Episode:
    return Episode(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        target_ids=torch.tensor([2, 3, 4], dtype=torch.long),
        reward=0.0,
        teacher_logits=torch.randn(3, 8),
        expert_ids=torch.tensor([step_id % 3], dtype=torch.long),
        router_probs=torch.ones(3) / 3.0,
        step_id=step_id,
    )


def test_replay_reservoir_has_reasonable_coverage() -> None:
    random.seed(0)
    buf = ReservoirReplayBuffer(capacity=100)
    for i in range(10_000):
        buf.add(_episode(i))

    assert len(buf) == 100
    mean_step = sum(ep.step_id for ep in buf._memory) / len(buf._memory)
    assert 3500 <= mean_step <= 6500


def test_sample_uniform_and_by_expert() -> None:
    buf = ReservoirReplayBuffer(capacity=20)
    for i in range(20):
        buf.add(_episode(i))

    sample = buf.sample_uniform(8)
    assert len(sample) == 8

    by_expert = buf.sample_by_expert(expert_id=1, n=10)
    assert len(by_expert) > 0
    assert all(1 in ep.expert_ids.tolist() for ep in by_expert)
