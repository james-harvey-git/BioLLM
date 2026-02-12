from __future__ import annotations

import torch

from biollm_cls.models.hippocampus import HippocampalMoE


def test_router_topk_and_probability_normalization() -> None:
    torch.manual_seed(0)
    moe = HippocampalMoE(hidden_size=16, vocab_size=32, num_experts=6, expert_hidden=8, top_k=2)
    hidden = torch.randn(4, 5, 16)
    _, routing = moe(hidden)

    assert routing.topk_indices.shape == (4, 2)
    assert routing.topk_probs.shape == (4, 2)
    sums = routing.topk_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_underused_experts_get_routing_bonus() -> None:
    torch.manual_seed(1)
    moe = HippocampalMoE(hidden_size=8, vocab_size=16, num_experts=4, expert_hidden=8, top_k=2, utilization_bonus=0.5)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
        moe.activation_counts[:] = torch.tensor([1000.0, 1.0, 1.0, 1.0])

    hidden = torch.zeros(2, 3, 8)
    _, routing = moe(hidden)
    mean_probs = routing.router_probs.mean(dim=0)
    assert mean_probs[0] < mean_probs[1]


def test_saturated_expert_gets_penalized() -> None:
    torch.manual_seed(2)
    moe = HippocampalMoE(hidden_size=8, vocab_size=16, num_experts=3, expert_hidden=8, top_k=1, capacity_penalty=1.0)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.zero_()
        moe.expert_drift[:] = torch.tensor([10.0, 0.1, 0.1])

    hidden = torch.zeros(1, 4, 8)
    _, routing = moe(hidden)
    probs = routing.router_probs[0]
    assert probs[0] < probs[1]


def test_apply_selected_handles_mixed_hidden_expert_dtypes() -> None:
    torch.manual_seed(3)
    moe = HippocampalMoE(hidden_size=8, vocab_size=16, num_experts=3, expert_hidden=8, top_k=1)
    # Experts remain float32 by default.
    hidden = torch.randn(2, 4, 8, dtype=torch.float16)
    expert_ids = torch.tensor([[0], [2]], dtype=torch.long)
    delta = moe.apply_selected(hidden, expert_ids)

    assert delta.shape == hidden.shape
    assert delta.dtype == hidden.dtype
