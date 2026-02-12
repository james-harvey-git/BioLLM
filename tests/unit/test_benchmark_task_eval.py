from __future__ import annotations

import torch

from biollm_cls.benchmarks.continual_toy import ContinualToyBenchmark
from biollm_cls.config import ModelConfig
from biollm_cls.models.neocortex import TinyCausalTransformer


def test_toy_evaluate_task_set_returns_all_requested_tasks() -> None:
    benchmark = ContinualToyBenchmark(vocab_size=32, seq_len=12, batch_size=4, switch_every=10, num_tasks=4, heldout_size=16)
    model = TinyCausalTransformer(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            seq_len=12,
            dropout=0.0,
            injection_layer=1,
        )
    )
    task_ids = [0, 2, 3]
    out = benchmark.evaluate_task_set(model, torch.device("cpu"), task_ids)

    assert sorted(out.keys()) == sorted(task_ids)
    for result in out.values():
        assert result.n_tokens > 0
        assert result.loss >= 0.0
        assert 0.0 <= result.token_acc <= 1.0
