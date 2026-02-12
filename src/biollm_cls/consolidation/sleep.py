from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from biollm_cls.consolidation.ewc import EWCState
from biollm_cls.memory.replay_buffer import Episode, ReservoirReplayBuffer
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
    pseudo_ratio: float = 0.25
    fisher_use_capability_mix: bool = True
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

    def _sample_pseudo_batch(self, replay_batch_size: int) -> list[Episode]:
        if not self.pseudo_rehearsal:
            return []
        pseudo_count = max(1, int(round(replay_batch_size * self.pseudo_ratio)))
        return self.replay_buffer.sample_uniform(pseudo_count)

    def _pseudo_teacher_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled and self.device.type == "cuda",
            ):
                hidden = self.base_model.forward_to_injection(input_ids)
                delta, _ = self.hippocampus(hidden, track_stats=False)
                full_logits = self.base_model.forward_from_injection(hidden + delta)
        return self._sanitize_logits(full_logits)

    def _shuffle_batch(self, input_ids: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape[0] <= 1:
            return input_ids, teacher_logits
        perm = torch.randperm(input_ids.shape[0], device=input_ids.device)
        return input_ids[perm], teacher_logits[perm]

    def run_sleep_cycle(self, steps: int) -> dict[str, float | str]:
        if len(self.replay_buffer) == 0 or steps <= 0:
            return {
                "steps": 0.0,
                "distill_loss": 0.0,
                "distill_kl": 0.0,
                "ewc_penalty": 0.0,
                "pseudo_batch_size": 0.0,
                "replay_batch_size_sleep": 0.0,
                "sleep_mix_ratio_pseudo": 0.0,
                "fisher_replay_samples": 0.0,
                "fisher_pseudo_samples": 0.0,
                "fisher_source_mode": "replay_only",
            }

        self.base_model.train()
        self.hippocampus.eval()

        total_distill = 0.0
        total_kl = 0.0
        total_ewc = 0.0
        total_pseudo = 0.0
        total_replay = 0.0
        total_mix_ratio_pseudo = 0.0
        total_fisher_replay = 0.0
        total_fisher_pseudo = 0.0
        fisher_source_mode = "replay_only"
        actual_steps = 0

        for _ in range(steps):
            replay_batch = self.replay_buffer.sample_uniform(self.replay_batch_size)
            if not replay_batch:
                break

            replay_input_ids = torch.stack([ep.input_ids for ep in replay_batch], dim=0).to(self.device)
            replay_teacher_logits = torch.stack([ep.teacher_logits for ep in replay_batch], dim=0).to(self.device)

            pseudo_batch = self._sample_pseudo_batch(len(replay_batch))
            pseudo_batch_size = len(pseudo_batch)

            mixed_input_ids = replay_input_ids
            mixed_teacher_logits = replay_teacher_logits
            if pseudo_batch_size > 0:
                pseudo_inputs = torch.stack([ep.input_ids for ep in pseudo_batch], dim=0).to(self.device)
                pseudo_teacher_logits = self._pseudo_teacher_logits(pseudo_inputs)
                mixed_input_ids = torch.cat([replay_input_ids, pseudo_inputs], dim=0)
                mixed_teacher_logits = torch.cat([replay_teacher_logits, pseudo_teacher_logits], dim=0)

            mixed_input_ids, mixed_teacher_logits = self._shuffle_batch(mixed_input_ids, mixed_teacher_logits)

            self.base_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled and self.device.type == "cuda",
            ):
                base_logits = self.base_model.forward_base(mixed_input_ids)
            distill_loss, kl = self._distill_loss(base_logits, mixed_teacher_logits)
            if not torch.isfinite(distill_loss):
                continue

            fisher_input_ids = replay_input_ids
            fisher_teacher_logits = replay_teacher_logits
            fisher_replay_samples = len(replay_batch)
            fisher_pseudo_samples = 0
            if self.fisher_use_capability_mix:
                fisher_input_ids = mixed_input_ids
                fisher_teacher_logits = mixed_teacher_logits
                fisher_pseudo_samples = pseudo_batch_size
                fisher_source_mode = "mixed_replay_pseudo" if pseudo_batch_size > 0 else "replay_only"
            else:
                fisher_source_mode = "replay_only"

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled and self.device.type == "cuda",
            ):
                fisher_logits = self.base_model.forward_base(fisher_input_ids)
            fisher_loss, _ = self._distill_loss(fisher_logits, fisher_teacher_logits)
            if not torch.isfinite(fisher_loss):
                continue

            self.ewc.update_fisher(self.base_model, fisher_loss)
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
            total_pseudo += float(pseudo_batch_size)
            total_replay += float(len(replay_batch))
            total_mix_ratio_pseudo += float(pseudo_batch_size / max(1, mixed_input_ids.shape[0]))
            total_fisher_replay += float(fisher_replay_samples)
            total_fisher_pseudo += float(fisher_pseudo_samples)
            actual_steps += 1

        self.ewc.snapshot_anchor(self.base_model)
        denom = max(actual_steps, 1)
        return {
            "steps": float(actual_steps),
            "distill_loss": total_distill / denom,
            "distill_kl": total_kl / denom,
            "ewc_penalty": total_ewc / denom,
            "pseudo_batch_size": total_pseudo / denom,
            "replay_batch_size_sleep": total_replay / denom,
            "sleep_mix_ratio_pseudo": total_mix_ratio_pseudo / denom,
            "fisher_replay_samples": total_fisher_replay / denom,
            "fisher_pseudo_samples": total_fisher_pseudo / denom,
            "fisher_source_mode": fisher_source_mode,
        }
