from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from biollm_cls.memory.replay_buffer import ReservoirReplayBuffer
from biollm_cls.models.base import NeocortexAdapter
from biollm_cls.models.hippocampus import HippocampalMoE


@dataclass
class ExpertRefresher:
    kl_threshold: float
    reset_factor: float
    samples_per_expert: int = 8

    def _expert_kl(
        self,
        base_model: NeocortexAdapter,
        hippocampus: HippocampalMoE,
        replay_buffer: ReservoirReplayBuffer,
        expert_id: int,
        device: torch.device,
    ) -> float:
        samples = replay_buffer.sample_by_expert(expert_id, self.samples_per_expert)
        if not samples:
            return float("inf")

        input_ids = torch.stack([s.input_ids for s in samples], dim=0).to(device)
        forced_ids = torch.full((input_ids.shape[0], 1), expert_id, device=device, dtype=torch.long)

        with torch.no_grad():
            hidden = base_model.forward_to_injection(input_ids)
            base_logits = base_model.forward_from_injection(hidden)
            delta = hippocampus.apply_selected(hidden, forced_ids)
            expert_logits = base_model.forward_from_injection(hidden + delta)
            kl = F.kl_div(
                torch.log_softmax(base_logits, dim=-1),
                torch.softmax(expert_logits, dim=-1),
                reduction="batchmean",
            )
        return float(kl.item())

    def run_refresh(
        self,
        base_model: NeocortexAdapter,
        hippocampus: HippocampalMoE,
        replay_buffer: ReservoirReplayBuffer,
        device: torch.device,
    ) -> dict[str, float]:
        if self.reset_factor >= 1.0:
            return {
                "refresh_count": 0.0,
                "refresh_avg_kl": 0.0,
            }

        refreshed = 0
        expert_kls: list[float] = []

        for expert_id in range(hippocampus.num_experts):
            kl = self._expert_kl(base_model, hippocampus, replay_buffer, expert_id, device)
            if kl == float("inf"):
                continue
            expert_kls.append(kl)
            if kl < self.kl_threshold:
                hippocampus.partial_reset_expert(expert_id, self.reset_factor)
                refreshed += 1

        avg_kl = float(sum(expert_kls) / max(1, len(expert_kls)))
        return {
            "refresh_count": float(refreshed),
            "refresh_avg_kl": avg_kl,
        }
