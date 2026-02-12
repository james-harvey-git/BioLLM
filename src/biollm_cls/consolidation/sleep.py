from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from biollm_cls.consolidation.ewc import EWCState
from biollm_cls.memory.replay_buffer import ReservoirReplayBuffer
from biollm_cls.models.base import NeocortexAdapter
from biollm_cls.models.hippocampus import HippocampalMoE


@dataclass
class Consolidator:
    base_model: NeocortexAdapter
    hippocampus: HippocampalMoE
    replay_buffer: ReservoirReplayBuffer
    ewc: EWCState
    base_optimizer: torch.optim.Optimizer
    device: torch.device
    replay_batch_size: int
    pseudo_rehearsal: bool = False
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16
    grad_scaler: torch.cuda.amp.GradScaler | None = None

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor, clamp: float = 30.0) -> torch.Tensor:
        return torch.nan_to_num(logits.float(), nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)

    def _distill_loss(self, base_logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_safe = self._sanitize_logits(teacher_logits)
        base_safe = self._sanitize_logits(base_logits)
        teacher_probs = torch.softmax(teacher_safe, dim=-1)
        base_log_probs = torch.log_softmax(base_safe, dim=-1)
        kl = F.kl_div(base_log_probs, teacher_probs, reduction="batchmean")
        ce = -(teacher_probs * base_log_probs).sum(dim=-1).mean()
        return 0.5 * (kl + ce), kl

    def run_sleep_cycle(self, steps: int) -> dict[str, float]:
        if len(self.replay_buffer) == 0 or steps <= 0:
            return {
                "steps": 0.0,
                "distill_loss": 0.0,
                "distill_kl": 0.0,
                "ewc_penalty": 0.0,
            }

        self.base_model.train()
        self.hippocampus.eval()

        total_distill = 0.0
        total_kl = 0.0
        total_ewc = 0.0
        actual_steps = 0

        for _ in range(steps):
            batch = self.replay_buffer.sample_uniform(self.replay_batch_size)
            if not batch:
                break

            input_ids = torch.stack([ep.input_ids for ep in batch], dim=0).to(self.device)
            teacher_logits = torch.stack([ep.teacher_logits for ep in batch], dim=0).to(self.device)

            self.base_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled and self.device.type == "cuda",
            ):
                base_logits = self.base_model.forward_base(input_ids)
            distill_loss, kl = self._distill_loss(base_logits, teacher_logits)
            if not torch.isfinite(distill_loss):
                continue
            self.ewc.update_fisher(self.base_model, distill_loss)
            ewc_pen = self.ewc.penalty(self.base_model)
            if not torch.isfinite(ewc_pen):
                continue
            total_loss = distill_loss + ewc_pen
            if not torch.isfinite(total_loss):
                continue
            if self.grad_scaler is not None and self.grad_scaler.is_enabled():
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.step(self.base_optimizer)
                self.grad_scaler.update()
            else:
                total_loss.backward()
                self.base_optimizer.step()

            total_distill += float(distill_loss.item())
            total_kl += float(kl.item())
            total_ewc += float(ewc_pen.item())
            actual_steps += 1

        self.ewc.snapshot_anchor(self.base_model)
        denom = max(actual_steps, 1)
        return {
            "steps": float(actual_steps),
            "distill_loss": total_distill / denom,
            "distill_kl": total_kl / denom,
            "ewc_penalty": total_ewc / denom,
        }
