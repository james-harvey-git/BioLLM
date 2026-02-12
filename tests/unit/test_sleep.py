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


def _populate_replay(model: TinyCausalTransformer, moe: HippocampalMoE, replay: ReservoirReplayBuffer, n: int = 16) -> None:
    for i in range(n):
        input_ids = torch.randint(0, 32, (8,), dtype=torch.long)
        target_ids = torch.randint(0, 32, (8,), dtype=torch.long)
        with torch.no_grad():
            hidden = model.forward_to_injection(input_ids.unsqueeze(0))
            delta, routing = moe(hidden)
            teacher_logits = model.forward_from_injection(hidden + delta)[0]
        replay.add(
            Episode(
                input_ids=input_ids,
                target_ids=target_ids,
                reward=0.0,
                teacher_logits=teacher_logits.detach().cpu(),
                expert_ids=routing.topk_indices[0].detach().cpu(),
                router_probs=routing.router_probs[0].detach().cpu(),
                step_id=i,
            )
        )


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


def test_sleep_distillation_reduces_base_teacher_kl_on_replay_subset() -> None:
    torch.manual_seed(12)
    model, moe, replay, ewc, optimizer = _make_components()
    ewc.lambda_ = 0.0

    _populate_replay(model, moe, replay, n=16)
    eval_batch = replay.sample_uniform(8)

    def _mean_kl(batch: list[Episode]) -> float:
        inputs = torch.stack([ep.input_ids for ep in batch], dim=0)
        teacher_logits = torch.stack([ep.teacher_logits for ep in batch], dim=0)
        with torch.no_grad():
            base_logits = model.forward_base(inputs)
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(base_logits, dim=-1),
                torch.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
        return float(kl.item())

    before = _mean_kl(eval_batch)
    consolidator = Consolidator(
        base_model=model,
        hippocampus=moe,
        replay_buffer=replay,
        ewc=ewc,
        base_optimizer=optimizer,
        device=torch.device("cpu"),
        replay_batch_size=8,
        amp_enabled=False,
        grad_scaler=None,
    )
    metrics = consolidator.run_sleep_cycle(steps=20)
    after = _mean_kl(eval_batch)

    assert metrics["steps"] > 0
    assert after <= before + 1e-6


def test_sleep_pseudo_rehearsal_generates_mixed_batch_without_stats_pollution() -> None:
    torch.manual_seed(17)
    model, moe, replay, ewc, optimizer = _make_components()
    _populate_replay(model, moe, replay, n=20)

    before_counts = moe.activation_counts.detach().clone()
    consolidator = Consolidator(
        base_model=model,
        hippocampus=moe,
        replay_buffer=replay,
        ewc=ewc,
        base_optimizer=optimizer,
        device=torch.device("cpu"),
        replay_batch_size=8,
        pseudo_rehearsal=True,
        pseudo_ratio=0.25,
        fisher_use_capability_mix=True,
        amp_enabled=False,
        grad_scaler=None,
    )
    metrics = consolidator.run_sleep_cycle(steps=4)

    assert metrics["steps"] > 0
    assert float(metrics["pseudo_batch_size"]) > 0.0
    assert float(metrics["sleep_mix_ratio_pseudo"]) > 0.0
    assert float(metrics["fisher_pseudo_samples"]) > 0.0
    assert metrics["fisher_source_mode"] == "mixed_replay_pseudo"
    assert torch.allclose(before_counts, moe.activation_counts)


def test_sleep_fisher_replay_only_when_pseudo_disabled() -> None:
    torch.manual_seed(19)
    model, moe, replay, ewc, optimizer = _make_components()
    _populate_replay(model, moe, replay, n=12)

    consolidator = Consolidator(
        base_model=model,
        hippocampus=moe,
        replay_buffer=replay,
        ewc=ewc,
        base_optimizer=optimizer,
        device=torch.device("cpu"),
        replay_batch_size=6,
        pseudo_rehearsal=False,
        fisher_use_capability_mix=True,
        amp_enabled=False,
        grad_scaler=None,
    )
    metrics = consolidator.run_sleep_cycle(steps=2)

    assert metrics["steps"] > 0
    assert float(metrics["fisher_pseudo_samples"]) == 0.0
    assert metrics["fisher_source_mode"] == "replay_only"
