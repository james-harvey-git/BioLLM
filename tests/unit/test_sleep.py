from __future__ import annotations

import math

import torch

from biollm_cls.config import ModelConfig
from biollm_cls.consolidation.ewc import EWCState
from biollm_cls.consolidation.sleep import Consolidator
from biollm_cls.memory.replay_buffer import Episode, ReservoirReplayBuffer
from biollm_cls.models.hippocampus import HippocampalMoE
from biollm_cls.models.neocortex import TinyCausalTransformer


def _make_components() -> tuple[TinyCausalTransformer, HippocampalMoE, ReservoirReplayBuffer, EWCState, torch.optim.AdamW]:
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
    replay = ReservoirReplayBuffer(capacity=32)
    ewc = EWCState(lambda_=10.0, fisher_decay=0.99, device=torch.device("cpu"))
    ewc.snapshot_anchor(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return model, moe, replay, ewc, optimizer


def test_sleep_cycle_handles_non_finite_teacher_logits() -> None:
    model, moe, replay, ewc, optimizer = _make_components()

    bad_logits = torch.randn(8, 32)
    bad_logits[0, 0] = float("inf")
    bad_logits[1, 1] = float("-inf")
    bad_logits[2, 2] = float("nan")

    ep = Episode(
        input_ids=torch.randint(0, 32, (8,), dtype=torch.long),
        target_ids=torch.randint(0, 32, (8,), dtype=torch.long),
        reward=0.0,
        teacher_logits=bad_logits,
        expert_ids=torch.tensor([0], dtype=torch.long),
        router_probs=torch.tensor([0.5, 0.5], dtype=torch.float32),
        step_id=1,
    )
    replay.add(ep)

    consolidator = Consolidator(
        base_model=model,
        hippocampus=moe,
        replay_buffer=replay,
        ewc=ewc,
        base_optimizer=optimizer,
        device=torch.device("cpu"),
        replay_batch_size=1,
        amp_enabled=False,
        grad_scaler=None,
    )
    metrics = consolidator.run_sleep_cycle(steps=2)

    assert metrics["steps"] >= 0.0
    for key in ("distill_loss", "distill_kl", "ewc_penalty"):
        assert math.isfinite(metrics[key])
