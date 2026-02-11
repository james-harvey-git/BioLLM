from __future__ import annotations

import torch

from biollm_cls.config import ModelConfig
from biollm_cls.consolidation.refresh import ExpertRefresher
from biollm_cls.memory.replay_buffer import Episode, ReservoirReplayBuffer
from biollm_cls.models.hippocampus import HippocampalMoE
from biollm_cls.models.neocortex import TinyCausalTransformer


def _episode_for_expert(expert_id: int) -> Episode:
    return Episode(
        input_ids=torch.randint(0, 32, (8,), dtype=torch.long),
        target_ids=torch.randint(0, 32, (8,), dtype=torch.long),
        reward=0.0,
        teacher_logits=torch.randn(8, 32),
        expert_ids=torch.tensor([expert_id], dtype=torch.long),
        router_probs=torch.ones(4) / 4.0,
        step_id=expert_id,
    )


def _make_model_and_moe() -> tuple[TinyCausalTransformer, HippocampalMoE]:
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        seq_len=8,
        dropout=0.0,
        injection_layer=1,
    )
    model = TinyCausalTransformer(cfg)
    moe = HippocampalMoE(hidden_size=16, vocab_size=32, num_experts=2, expert_hidden=8, top_k=1)
    return model, moe


def test_refresh_resets_low_kl_expert() -> None:
    torch.manual_seed(0)
    model, moe = _make_model_and_moe()
    buf = ReservoirReplayBuffer(capacity=16)
    for _ in range(8):
        buf.add(_episode_for_expert(0))

    with torch.no_grad():
        for p in moe.experts[0].parameters():
            p.add_(0.01)

    before = [p.detach().clone() for p in moe.experts[0].parameters()]
    refresher = ExpertRefresher(kl_threshold=1.0, reset_factor=0.1, samples_per_expert=4)
    out = refresher.run_refresh(model, moe, buf, torch.device("cpu"))
    after = [p.detach().clone() for p in moe.experts[0].parameters()]

    assert out["refresh_count"] >= 1
    assert any((a - b).abs().sum().item() > 0 for a, b in zip(before, after))


def test_refresh_preserves_high_kl_expert_when_threshold_tiny() -> None:
    torch.manual_seed(1)
    model, moe = _make_model_and_moe()
    buf = ReservoirReplayBuffer(capacity=16)
    for _ in range(8):
        buf.add(_episode_for_expert(1))

    with torch.no_grad():
        for p in moe.experts[1].parameters():
            p.add_(5.0)

    before = [p.detach().clone() for p in moe.experts[1].parameters()]
    refresher = ExpertRefresher(kl_threshold=1e-8, reset_factor=0.1, samples_per_expert=4)
    out = refresher.run_refresh(model, moe, buf, torch.device("cpu"))
    after = [p.detach().clone() for p in moe.experts[1].parameters()]

    assert out["refresh_count"] == 0
    assert all(torch.allclose(a, b) for a, b in zip(before, after))
