from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RoutingInfo:
    topk_indices: torch.Tensor
    topk_probs: torch.Tensor
    router_probs: torch.Tensor


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HippocampalMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_experts: int = 32,
        expert_hidden: int = 128,
        top_k: int = 2,
        utilization_bonus: float = 0.1,
        capacity_penalty: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.utilization_bonus = utilization_bonus
        self.capacity_penalty = capacity_penalty

        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size, expert_hidden) for _ in range(num_experts)])
        self.pred_head = nn.Linear(hidden_size, vocab_size)

        self.register_buffer("activation_counts", torch.zeros(num_experts))
        self.register_buffer("expert_drift", torch.zeros(num_experts))
        self.register_buffer("last_router_probs", torch.zeros(num_experts))

        self._expert_init: list[dict[str, torch.Tensor]] = []
        for expert in self.experts:
            self._expert_init.append({k: v.detach().clone() for k, v in expert.state_dict().items()})

    def _usage_bonus_vector(self) -> torch.Tensor:
        if float(self.activation_counts.sum().item()) <= 0.0:
            return torch.zeros_like(self.activation_counts)
        usage = self.activation_counts / self.activation_counts.sum().clamp_min(1e-8)
        bonus = self.utilization_bonus * (1.0 - usage)
        return bonus

    def capacity_scores(self) -> torch.Tensor:
        usage = self.activation_counts
        usage_norm = usage / usage.max().clamp_min(1e-8)
        drift_norm = self.expert_drift / self.expert_drift.max().clamp_min(1e-8)
        return (usage_norm + drift_norm) / 2.0

    def _effective_router_logits(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        base_logits = self.router(pooled_hidden)
        bonus = self._usage_bonus_vector().unsqueeze(0)
        penalty = self.capacity_penalty * self.capacity_scores().unsqueeze(0)
        return base_logits + bonus - penalty

    def _apply_topk(self, hidden: torch.Tensor, topk_idx: torch.Tensor, topk_probs: torch.Tensor) -> torch.Tensor:
        bsz, _, hdim = hidden.shape
        delta = torch.zeros_like(hidden)
        for b in range(bsz):
            for j in range(self.top_k):
                eid = int(topk_idx[b, j].item())
                weight = topk_probs[b, j]
                delta[b] = delta[b] + weight * self.experts[eid](hidden[b])
                self.activation_counts[eid] += 1.0
        assert delta.shape[-1] == hdim
        return delta

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, RoutingInfo]:
        pooled = hidden.mean(dim=1)
        effective_logits = self._effective_router_logits(pooled)
        router_probs = torch.softmax(effective_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        delta = self._apply_topk(hidden, topk_idx, topk_probs)
        self.last_router_probs = router_probs.detach().mean(dim=0)
        return delta, RoutingInfo(topk_indices=topk_idx, topk_probs=topk_probs, router_probs=router_probs)

    def apply_selected(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a forced expert selection, used for expert refresh checks."""
        if expert_weights is None:
            expert_weights = torch.ones_like(expert_ids, dtype=hidden.dtype)
        if expert_ids.dim() == 1:
            expert_ids = expert_ids.unsqueeze(-1)
            expert_weights = expert_weights.unsqueeze(-1)

        bsz = hidden.shape[0]
        delta = torch.zeros_like(hidden)
        for b in range(bsz):
            for j in range(expert_ids.shape[1]):
                eid = int(expert_ids[b, j].item())
                w = expert_weights[b, j]
                delta[b] = delta[b] + w * self.experts[eid](hidden[b])
        return delta

    def predict_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.pred_head(hidden)

    def routing_regularizer(self) -> torch.Tensor:
        target = torch.full_like(self.last_router_probs, 1.0 / self.num_experts)
        return ((self.last_router_probs - target) ** 2).mean()

    def update_capacity_estimates(self) -> None:
        drifts = []
        for expert_idx, expert in enumerate(self.experts):
            d = 0.0
            cur_state = expert.state_dict()
            for key, value in cur_state.items():
                init_val = self._expert_init[expert_idx][key].to(value.device)
                d += torch.norm(value.detach() - init_val, p=2).item()
            drifts.append(d)
        self.expert_drift = torch.tensor(drifts, device=self.expert_drift.device, dtype=self.expert_drift.dtype)

    def partial_reset_expert(self, expert_id: int, factor: float) -> None:
        expert = self.experts[expert_id]
        with torch.no_grad():
            cur_state = expert.state_dict()
            blended = {}
            for key, value in cur_state.items():
                init_val = self._expert_init[expert_id][key].to(value.device)
                blended[key] = init_val + factor * (value - init_val)
            expert.load_state_dict(blended)

            if self.router.bias is not None:
                self.router.bias[expert_id].mul_(factor)
            self.activation_counts[expert_id].mul_(factor)
            self.expert_drift[expert_id].mul_(factor)

    def utilization_entropy(self) -> float:
        p = self.activation_counts / self.activation_counts.sum().clamp_min(1e-8)
        entropy = -(p * (p + 1e-8).log()).sum()
        return float(entropy.item())
