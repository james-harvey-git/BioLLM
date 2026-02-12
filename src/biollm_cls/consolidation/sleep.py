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
    pseudo_source: str = "mixed"
    pseudo_random_fraction: float = 0.5
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

    def _normalize_pseudo_source(self) -> str:
        source = self.pseudo_source.strip().lower()
        if source in {"replay", "replay_echo", "replay-input-echo"}:
            return "replay_echo"
        if source in {"random", "random_tokens", "random-token-prompts"}:
            return "random_tokens"
        return "mixed"

    def _pseudo_counts(self, replay_batch_size: int) -> tuple[int, int]:
        if not self.pseudo_rehearsal:
            return 0, 0

        pseudo_count = max(1, int(round(replay_batch_size * self.pseudo_ratio)))
        source = self._normalize_pseudo_source()

        if source == "replay_echo":
            return pseudo_count, 0
        if source == "random_tokens":
            return 0, pseudo_count

        random_fraction = float(min(1.0, max(0.0, self.pseudo_random_fraction)))
        random_count = int(round(pseudo_count * random_fraction))
        random_count = max(0, min(pseudo_count, random_count))
        replay_count = pseudo_count - random_count

        # Keep mixed mode actually mixed when there is enough budget.
        if pseudo_count >= 2 and replay_count == 0:
            replay_count, random_count = 1, pseudo_count - 1
        if pseudo_count >= 2 and random_count == 0:
            random_count, replay_count = 1, pseudo_count - 1
        return replay_count, random_count

    def _sample_random_inputs(self, n: int, seq_len: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0, seq_len), device=self.device, dtype=torch.long)
        vocab_cap = max(2, int(getattr(self.base_model, "vocab_size", 2)))
        return torch.randint(low=0, high=vocab_cap, size=(n, seq_len), device=self.device, dtype=torch.long)

    def _sample_replay_pseudo_inputs(self, n: int, seq_len: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0, seq_len), device=self.device, dtype=torch.long)
        sampled = self.replay_buffer.sample_uniform(n)
        if not sampled:
            return torch.empty((0, seq_len), device=self.device, dtype=torch.long)
        return torch.stack([ep.input_ids for ep in sampled], dim=0).to(self.device)

    def _build_pseudo_inputs(self, replay_batch: list[Episode], replay_input_ids: torch.Tensor) -> torch.Tensor:
        replay_count, random_count = self._pseudo_counts(len(replay_batch))
        seq_len = int(replay_input_ids.shape[1])

        replay_inputs = self._sample_replay_pseudo_inputs(replay_count, seq_len)
        random_inputs = self._sample_random_inputs(random_count, seq_len)

        if replay_inputs.shape[0] <= 0 and random_inputs.shape[0] <= 0:
            return torch.empty((0, seq_len), device=self.device, dtype=torch.long)
        if replay_inputs.shape[0] <= 0:
            return random_inputs
        if random_inputs.shape[0] <= 0:
            return replay_inputs
        return torch.cat([replay_inputs, random_inputs], dim=0)

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
                "pseudo_source_mode": self._normalize_pseudo_source(),
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
        pseudo_source_mode = self._normalize_pseudo_source()
        actual_steps = 0

        for _ in range(steps):
            replay_batch = self.replay_buffer.sample_uniform(self.replay_batch_size)
            if not replay_batch:
                break

            replay_input_ids = torch.stack([ep.input_ids for ep in replay_batch], dim=0).to(self.device)
            replay_teacher_logits = torch.stack([ep.teacher_logits for ep in replay_batch], dim=0).to(self.device)

            pseudo_inputs = self._build_pseudo_inputs(replay_batch, replay_input_ids)
            pseudo_batch_size = int(pseudo_inputs.shape[0])

            mixed_input_ids = replay_input_ids
            mixed_teacher_logits = replay_teacher_logits
            if pseudo_batch_size > 0:
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

            fisher_replay_samples = len(replay_batch)
            fisher_pseudo_samples = 0
            reuse_distill_for_fisher = False

            if self.fisher_use_capability_mix:
                fisher_source_mode = "mixed_replay_pseudo" if pseudo_batch_size > 0 else "replay_only"
                fisher_pseudo_samples = pseudo_batch_size
                reuse_distill_for_fisher = True
            else:
                fisher_source_mode = "replay_only"
                reuse_distill_for_fisher = pseudo_batch_size == 0

            if reuse_distill_for_fisher:
                fisher_loss = distill_loss
            else:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled and self.device.type == "cuda",
                ):
                    fisher_logits = self.base_model.forward_base(replay_input_ids)
                fisher_loss, _ = self._distill_loss(fisher_logits, replay_teacher_logits)
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
            "pseudo_source_mode": pseudo_source_mode,
        }
