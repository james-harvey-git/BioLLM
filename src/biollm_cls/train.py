from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from biollm_cls.benchmarks.factory import build_benchmark
from biollm_cls.config import CLSConfig
from biollm_cls.consolidation.ewc import EWCState
from biollm_cls.consolidation.refresh import ExpertRefresher
from biollm_cls.consolidation.sleep import Consolidator
from biollm_cls.control.scheduler import SleepPressureController, StepStats
from biollm_cls.logging import MetricsLogger
from biollm_cls.memory.replay_buffer import Episode, ReservoirReplayBuffer
from biollm_cls.models.factory import build_neocortex
from biollm_cls.models.hippocampus import HippocampalMoE
from biollm_cls.repro import save_run_metadata, set_seed


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower().strip()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype '{dtype_name}'. Use one of: fp16, bf16")


def _backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def _optimizer_has_fp16_params(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is None:
                continue
            if getattr(param, "dtype", None) == torch.float16:
                return True
    return False


def run_training(cfg: CLSConfig) -> dict[str, float]:
    device = _resolve_device(cfg.device)
    set_seed(cfg.seed, cfg.repro.deterministic)

    if cfg.train.matmul_precision:
        try:
            torch.set_float32_matmul_precision(cfg.train.matmul_precision)
        except Exception:
            pass

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.allow_tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(cfg.train.allow_tf32)

    amp_dtype = _resolve_amp_dtype(cfg.train.amp_dtype)
    amp_enabled = bool(cfg.train.amp_enabled and device.type == "cuda")

    model = build_neocortex(cfg.model).to(device)
    runtime_vocab_size = int(min(cfg.model.vocab_size, model.vocab_size))
    hippocampus = HippocampalMoE(
        hidden_size=model.hidden_size,
        vocab_size=runtime_vocab_size,
        num_experts=cfg.experts.num_experts,
        expert_hidden=cfg.experts.expert_hidden,
        top_k=cfg.experts.top_k,
        utilization_bonus=cfg.experts.utilization_bonus,
        capacity_penalty=cfg.experts.capacity_penalty,
    ).to(device)

    if cfg.train.compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.train.compile_mode)
        hippocampus = torch.compile(hippocampus, mode=cfg.train.compile_mode)

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.consolidation.slow_lr)
    fast_optimizer = torch.optim.AdamW(hippocampus.parameters(), lr=cfg.experts.fast_lr)

    # GradScaler cannot unscale FP16 gradients from FP16 parameters.
    # HF models loaded with model.hf_torch_dtype=fp16 have FP16 params, so disable scaler there.
    use_grad_scaler = bool(amp_enabled and amp_dtype == torch.float16)
    use_base_scaler = bool(use_grad_scaler and not _optimizer_has_fp16_params(base_optimizer))
    use_fast_scaler = bool(use_grad_scaler and not _optimizer_has_fp16_params(fast_optimizer))
    fast_scaler = torch.cuda.amp.GradScaler(enabled=use_fast_scaler) if use_fast_scaler else None
    base_scaler = torch.cuda.amp.GradScaler(enabled=use_base_scaler) if use_base_scaler else None

    replay = ReservoirReplayBuffer(cfg.replay.capacity)
    ewc = EWCState(cfg.ewc.lambda_, cfg.ewc.fisher_decay, device=device)
    ewc.snapshot_anchor(model)

    benchmark = build_benchmark(
        benchmark_cfg=cfg.benchmark,
        model_cfg=cfg.model,
        runtime_vocab_size=runtime_vocab_size,
        seq_len=cfg.model.seq_len,
        batch_size=cfg.train.batch_size,
        seed=cfg.seed,
    )

    consolidator = Consolidator(
        base_model=model,
        hippocampus=hippocampus,
        replay_buffer=replay,
        ewc=ewc,
        base_optimizer=base_optimizer,
        device=device,
        replay_batch_size=cfg.replay.batch_size,
        pseudo_rehearsal=cfg.consolidation.pseudo_rehearsal,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        grad_scaler=base_scaler,
    )
    refresher = ExpertRefresher(
        kl_threshold=cfg.consolidation.refresh.kl_threshold,
        reset_factor=cfg.consolidation.refresh.reset_factor,
    )
    scheduler = SleepPressureController(
        saturation_weight=cfg.scheduler.saturation_weight,
        forgetting_weight=cfg.scheduler.forgetting_weight,
        novelty_weight=cfg.scheduler.novelty_weight,
        pressure_threshold=cfg.scheduler.pressure_threshold,
        min_interval=cfg.consolidation.sleep_min_interval,
        max_interval=cfg.consolidation.sleep_max_interval,
        min_sleep_steps=cfg.consolidation.min_sleep_steps,
        max_sleep_steps=cfg.consolidation.max_sleep_steps,
    )

    config_dict = asdict(cfg)
    config_dict["runtime"] = {
        "resolved_device": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": cfg.train.amp_dtype,
        "base_grad_scaler_enabled": use_base_scaler,
        "fast_grad_scaler_enabled": use_fast_scaler,
        "runtime_vocab_size": runtime_vocab_size,
        "model_provider": cfg.model.provider,
    }
    logger = MetricsLogger(
        output_dir=cfg.logging.output_dir,
        use_wandb=cfg.logging.use_wandb,
        wandb_project=cfg.logging.wandb_project,
        wandb_mode=cfg.logging.wandb_mode,
        config=config_dict,
        wandb_entity=cfg.logging.wandb_entity,
        wandb_run_name=cfg.logging.wandb_run_name,
        wandb_group=cfg.logging.wandb_group,
        wandb_job_type=cfg.logging.wandb_job_type,
        wandb_tags=cfg.logging.wandb_tags,
        wandb_notes=cfg.logging.wandb_notes,
        wandb_resume=cfg.logging.wandb_resume,
        wandb_run_id=cfg.logging.wandb_run_id,
        wandb_api_key_env_var=cfg.logging.wandb_api_key_env_var,
        checkpoint_keep_last=cfg.logging.checkpoint_keep_last,
        upload_checkpoints=cfg.logging.upload_checkpoints,
        upload_config_artifact=cfg.logging.upload_config_artifact,
        upload_metadata_artifact=cfg.logging.upload_metadata_artifact,
    )

    if cfg.logging.wandb_watch_model:
        logger.watch(model, log_freq=cfg.logging.wandb_watch_log_freq)

    if cfg.repro.save_config_snapshot:
        save_run_metadata(Path(cfg.logging.output_dir), config_dict)
    if cfg.logging.upload_config_artifact:
        logger.log_artifact_file(
            Path(cfg.logging.output_dir) / "validated_config.json",
            artifact_name="validated-config",
            artifact_type="config",
        )
    if cfg.logging.upload_metadata_artifact:
        logger.log_artifact_file(
            Path(cfg.logging.output_dir) / "run_metadata.json",
            artifact_name="run-metadata",
            artifact_type="metadata",
        )

    used_experts: set[int] = set()
    best_old_loss = float("inf")
    old_loss = 0.0
    sleep_count = 0
    refresh_count_total = 0
    sleep_base_updates = 0

    try:
        for step in range(1, cfg.train.max_steps + 1):
            batch = benchmark.sample_batch(step, cfg.train.batch_size)
            input_ids = batch.input_ids.to(device)
            target_ids = batch.target_ids.to(device)

            wake_base_enabled = cfg.consolidation.wake_base_lr_scale > 0.0

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if wake_base_enabled:
                    hidden = model.forward_to_injection(input_ids)
                else:
                    with torch.no_grad():
                        hidden = model.forward_to_injection(input_ids)

                delta, routing = hippocampus(hidden.detach() if not wake_base_enabled else hidden)
                augmented_hidden = hidden + delta

                fast_logits = hippocampus.predict_logits(augmented_hidden)
                fast_ce = F.cross_entropy(
                    fast_logits.view(-1, runtime_vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100,
                )
                fast_loss = fast_ce + cfg.experts.routing_reg_weight * hippocampus.routing_regularizer()

            reward = 0.0
            if cfg.train.reward_from_correctness:
                reward = benchmark.reward_from_correctness(fast_logits, target_ids)
            reward_mult = float(np.clip(1.0 + cfg.experts.reward_alpha * reward, 0.25, 2.0))

            fast_optimizer.zero_grad(set_to_none=True)
            _backward_step(fast_loss * reward_mult, fast_optimizer, fast_scaler)
            hippocampus.update_capacity_estimates()

            if wake_base_enabled:
                base_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    wake_logits = model.forward_from_injection(augmented_hidden)
                    wake_loss = F.cross_entropy(
                        wake_logits.view(-1, runtime_vocab_size),
                        target_ids.view(-1),
                        ignore_index=-100,
                    )
                _backward_step(wake_loss * cfg.consolidation.wake_base_lr_scale, base_optimizer, base_scaler)
                teacher_logits = wake_logits.detach()
            else:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        teacher_logits = model.forward_from_injection(augmented_hidden.detach())

            episode = Episode(
                input_ids=input_ids[0].detach().cpu(),
                target_ids=target_ids[0].detach().cpu(),
                reward=float(reward),
                teacher_logits=teacher_logits[0].detach().cpu().to(torch.float16),
                expert_ids=routing.topk_indices[0].detach().cpu(),
                router_probs=routing.router_probs[0].detach().cpu(),
                step_id=step,
            )
            replay.add(episode)

            selected = set(int(v) for v in routing.topk_indices.detach().cpu().view(-1).tolist())
            novelty_rate = len([x for x in selected if x not in used_experts]) / max(1, len(selected))
            used_experts.update(selected)

            saturation_scores = hippocampus.capacity_scores().detach().cpu()
            saturation_ratio = float((saturation_scores > cfg.experts.saturation_threshold).float().mean().item())

            if step % cfg.train.eval_interval == 0:
                current_task = benchmark.current_task(step)
                old_loss = benchmark.evaluate_old_loss(model, device, current_task=current_task)
                best_old_loss = min(best_old_loss, old_loss)
            forgetting_signal = max(0.0, old_loss - best_old_loss) if best_old_loss < float("inf") else 0.0

            scheduler.update(
                StepStats(
                    saturation_ratio=saturation_ratio,
                    forgetting_signal=forgetting_signal,
                    novelty_rate=novelty_rate,
                )
            )

            sleep_metrics = {
                "distill_loss": 0.0,
                "distill_kl": 0.0,
                "ewc_penalty": 0.0,
                "steps": 0.0,
            }
            refresh_metrics = {"refresh_count": 0.0, "refresh_avg_kl": 0.0}
            if not cfg.train.ablation_no_sleep and scheduler.should_sleep(step):
                sleep_steps = scheduler.recommended_sleep_steps()
                sleep_metrics = consolidator.run_sleep_cycle(sleep_steps)
                refresh_metrics = refresher.run_refresh(model, hippocampus, replay, device)
                scheduler.mark_slept(step)
                sleep_count += 1
                sleep_base_updates += int(sleep_metrics["steps"])
                refresh_count_total += int(refresh_metrics["refresh_count"])

            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    base_logits = model.forward_base(input_ids[:1])
                teacher_probs = torch.softmax(teacher_logits[:1].float(), dim=-1)
                base_teacher_kl = float(
                    F.kl_div(
                        torch.log_softmax(base_logits.float(), dim=-1),
                        teacher_probs,
                        reduction="batchmean",
                    ).item()
                )

            perf_metrics = {}
            if device.type == "cuda":
                perf_metrics["cuda_mem_alloc_mb"] = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
                perf_metrics["cuda_mem_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024 ** 2))

            logger.log(
                {
                    "task_loss": float(fast_ce.item()),
                    "old_task_loss": float(old_loss),
                    "forgetting_index": float(forgetting_signal),
                    "sleep_pressure": float(scheduler.current_pressure),
                    "sleep_count": float(sleep_count),
                    "sleep_steps": float(sleep_metrics["steps"]),
                    "expert_util_entropy": float(hippocampus.utilization_entropy()),
                    "refresh_count": float(refresh_metrics["refresh_count"]),
                    "base_teacher_kl": float(base_teacher_kl),
                    "distill_kl": float(sleep_metrics["distill_kl"]),
                    "ewc_penalty": float(sleep_metrics["ewc_penalty"]),
                    "replay_size": float(len(replay)),
                    **perf_metrics,
                },
                step=step,
            )

            if cfg.logging.checkpoint_interval > 0 and step % cfg.logging.checkpoint_interval == 0:
                logger.save_checkpoint(
                    step=step,
                    model=model,
                    hippocampus=hippocampus,
                    base_optimizer=base_optimizer,
                    fast_optimizer=fast_optimizer,
                    extra={
                        "sleep_count": sleep_count,
                        "refresh_count_total": refresh_count_total,
                        "resolved_device": str(device),
                    },
                )

        final_summary = {
            "sleep_count": float(sleep_count),
            "refresh_count_total": float(refresh_count_total),
            "base_updates_in_sleep": float(sleep_base_updates),
            "final_forgetting_index": float(max(0.0, old_loss - best_old_loss) if best_old_loss < float("inf") else 0.0),
            "final_old_task_loss": float(old_loss),
        }
        logger.save_checkpoint(
            step=cfg.train.max_steps,
            model=model,
            hippocampus=hippocampus,
            base_optimizer=base_optimizer,
            fast_optimizer=fast_optimizer,
            extra={**final_summary, "resolved_device": str(device)},
        )
        logger.log_summary(final_summary)
        return final_summary
    finally:
        logger.close()
