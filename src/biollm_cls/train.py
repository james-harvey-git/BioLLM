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
from biollm_cls.eval.continual_metrics import ContinualMetricsTracker
from biollm_cls.logging import MetricsLogger
from biollm_cls.memory.replay_buffer import Episode, ReservoirReplayBuffer
from biollm_cls.metrics_catalog import render_metric_glossary_markdown, training_metric_glossary_rows
from biollm_cls.models.factory import build_neocortex
from biollm_cls.models.hippocampus import HippocampalMoE, RoutingInfo
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


def _sanitize_logits(logits: torch.Tensor, clamp: float = 30.0) -> torch.Tensor:
    return torch.nan_to_num(logits.float(), nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)


def _safe_token_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, int]:
    mask = targets != -100
    supervised_tokens = int(mask.sum().item())
    if supervised_tokens <= 0:
        # Keep graph connectivity with zero gradient contribution.
        return logits.sum() * 0.0, 0
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100,
    )
    return loss, supervised_tokens


def _optimizer_has_fp16_params(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is None:
                continue
            if getattr(param, "dtype", None) == torch.float16:
                return True
    return False


def _neutral_routing(batch_size: int, top_k: int, num_experts: int, device: torch.device) -> RoutingInfo:
    safe_topk = max(1, int(top_k))
    safe_experts = max(1, int(num_experts))
    row_idx = torch.arange(safe_topk, device=device, dtype=torch.long) % safe_experts
    topk_indices = row_idx.unsqueeze(0).repeat(batch_size, 1)
    topk_probs = torch.full((batch_size, safe_topk), 1.0 / safe_topk, device=device)
    router_probs = torch.full((batch_size, safe_experts), 1.0 / safe_experts, device=device)
    return RoutingInfo(topk_indices=topk_indices, topk_probs=topk_probs, router_probs=router_probs)


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
        pseudo_ratio=cfg.consolidation.pseudo_ratio,
        fisher_use_capability_mix=cfg.consolidation.fisher_use_capability_mix,
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
    cl_metrics = ContinualMetricsTracker()

    config_dict = asdict(cfg)
    runtime_config = {
        "resolved_device": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": cfg.train.amp_dtype,
        "base_grad_scaler_enabled": use_base_scaler,
        "fast_grad_scaler_enabled": use_fast_scaler,
        "runtime_vocab_size": runtime_vocab_size,
        "model_provider": cfg.model.provider,
    }
    if hasattr(model, "runtime_info") and callable(getattr(model, "runtime_info")):
        try:
            runtime_config.update(dict(model.runtime_info()))
        except Exception:
            pass
    config_dict["runtime"] = runtime_config
    metric_glossary_rows = training_metric_glossary_rows()
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
        metric_glossary_rows=metric_glossary_rows,
        metric_glossary_markdown=render_metric_glossary_markdown(
            "Training W&B Metrics",
            metric_glossary_rows,
        ),
    )

    if cfg.logging.wandb_watch_model:
        logger.watch(model, log_freq=cfg.logging.wandb_watch_log_freq)

    hf_runtime_keys = (
        "hf_injection_mode",
        "hf_injection_layer_idx",
        "hf_num_decoder_layers",
        "hf_injection_fraction",
    )
    hf_runtime_payload = {k: runtime_config[k] for k in hf_runtime_keys if k in runtime_config}
    if hf_runtime_payload:
        logger.log(hf_runtime_payload, step=0)

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

    disable_hippocampus = bool(cfg.train.ablation_disable_hippocampus)
    refresh_enabled = (not disable_hippocampus) and (cfg.consolidation.refresh.reset_factor < 1.0)

    used_experts: set[int] = set()
    seen_tasks: set[int] = set()
    previous_task_id: int | None = None

    best_old_loss = float("inf")
    old_loss = 0.0
    sleep_count = 0
    refresh_count_total = 0
    sleep_base_updates = 0

    try:
        for step in range(1, cfg.train.max_steps + 1):
            scheduled_task = benchmark.current_task(step)
            if previous_task_id is not None and scheduled_task != previous_task_id:
                boundary_eval = benchmark.evaluate_task_set(model, device, sorted(seen_tasks))
                boundary_metrics = cl_metrics.record_boundary(
                    step=step - 1,
                    completed_task_id=previous_task_id,
                    task_results=boundary_eval,
                )
                boundary_payload = {
                    "seen_acc_avg": float(boundary_metrics["seen_acc_avg"]),
                    "seen_loss_avg": float(boundary_metrics["seen_loss_avg"]),
                    "forgetting": float(boundary_metrics["forgetting"]),
                    "bwt": float(boundary_metrics["bwt"]),
                    "tasks_completed": float(boundary_metrics["tasks_completed"]),
                    "boundary_index": float(boundary_metrics["boundary_index"]),
                    "boundary_step": float(boundary_metrics["boundary_step"]),
                }
                for task_id, result in boundary_eval.items():
                    boundary_payload[f"task_acc/{task_id}"] = float(result.token_acc)
                    boundary_payload[f"task_loss/{task_id}"] = float(result.loss)
                    boundary_payload[f"task_tokens/{task_id}"] = float(result.n_tokens)
                logger.log(boundary_payload, step=step - 1)

            batch = benchmark.sample_batch(step, cfg.train.batch_size)
            current_task = int(batch.task_id)
            seen_tasks.add(current_task)

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

                if disable_hippocampus:
                    delta = torch.zeros_like(hidden)
                    routing = _neutral_routing(
                        batch_size=hidden.shape[0],
                        top_k=cfg.experts.top_k,
                        num_experts=cfg.experts.num_experts,
                        device=hidden.device,
                    )
                    augmented_hidden = hidden
                    fast_logits = model.forward_from_injection(augmented_hidden.detach() if not wake_base_enabled else augmented_hidden)
                    fast_ce, batch_supervised_tokens = _safe_token_ce(fast_logits, target_ids, runtime_vocab_size)
                    fast_loss = fast_ce
                else:
                    delta, routing = hippocampus(hidden.detach() if not wake_base_enabled else hidden)
                    augmented_hidden = hidden + delta
                    fast_logits = hippocampus.predict_logits(augmented_hidden)
                    fast_ce, batch_supervised_tokens = _safe_token_ce(fast_logits, target_ids, runtime_vocab_size)
                    fast_loss = fast_ce + cfg.experts.routing_reg_weight * hippocampus.routing_regularizer()

            reward = 0.0
            if cfg.train.reward_from_correctness:
                reward = benchmark.reward_from_correctness(fast_logits, target_ids)
            reward_mult = float(np.clip(1.0 + cfg.experts.reward_alpha * reward, 0.25, 2.0))

            fast_update_applied = 0.0
            if not disable_hippocampus:
                fast_optimizer.zero_grad(set_to_none=True)
                _backward_step(fast_loss * reward_mult, fast_optimizer, fast_scaler)
                hippocampus.update_capacity_estimates()
                fast_update_applied = 1.0

            if wake_base_enabled:
                base_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    wake_logits = model.forward_from_injection(augmented_hidden)
                    wake_loss, _ = _safe_token_ce(wake_logits, target_ids, runtime_vocab_size)
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

            for row_idx in range(input_ids.shape[0]):
                episode = Episode(
                    input_ids=input_ids[row_idx].detach().cpu(),
                    target_ids=target_ids[row_idx].detach().cpu(),
                    reward=float(reward),
                    teacher_logits=_sanitize_logits(teacher_logits[row_idx].detach().cpu(), clamp=30.0).to(torch.float16),
                    expert_ids=routing.topk_indices[row_idx].detach().cpu(),
                    router_probs=routing.router_probs[row_idx].detach().cpu(),
                    step_id=step,
                )
                replay.add(episode)
            replay_added_this_step = int(input_ids.shape[0])

            if disable_hippocampus:
                novelty_rate = 0.0
                saturation_ratio = 0.0
                expert_util_entropy = 0.0
            else:
                selected = set(int(v) for v in routing.topk_indices.detach().cpu().view(-1).tolist())
                novelty_rate = len([x for x in selected if x not in used_experts]) / max(1, len(selected))
                used_experts.update(selected)

                saturation_scores = hippocampus.capacity_scores().detach().cpu()
                saturation_ratio = float((saturation_scores > cfg.experts.saturation_threshold).float().mean().item())
                expert_util_entropy = float(hippocampus.utilization_entropy())

            if step % cfg.train.eval_interval == 0:
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
                "pseudo_batch_size": 0.0,
                "replay_batch_size_sleep": 0.0,
                "sleep_mix_ratio_pseudo": 0.0,
                "fisher_replay_samples": 0.0,
                "fisher_pseudo_samples": 0.0,
                "fisher_source_mode": "replay_only",
            }
            refresh_metrics = {"refresh_count": 0.0, "refresh_avg_kl": 0.0}
            if not cfg.train.ablation_no_sleep and scheduler.should_sleep(step):
                sleep_steps = scheduler.recommended_sleep_steps()
                sleep_metrics = consolidator.run_sleep_cycle(sleep_steps)
                if refresh_enabled:
                    refresh_metrics = refresher.run_refresh(model, hippocampus, replay, device)
                    refresh_count_total += int(refresh_metrics["refresh_count"])
                scheduler.mark_slept(step)
                sleep_count += 1
                sleep_base_updates += int(sleep_metrics["steps"])

            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    base_logits = model.forward_base(input_ids[:1])
                teacher_probs = torch.softmax(_sanitize_logits(teacher_logits[:1], clamp=30.0), dim=-1)
                base_log_probs = torch.log_softmax(_sanitize_logits(base_logits, clamp=30.0), dim=-1)
                base_teacher_kl_t = F.kl_div(base_log_probs, teacher_probs, reduction="batchmean")
                base_teacher_kl = float(base_teacher_kl_t.item()) if torch.isfinite(base_teacher_kl_t) else float("nan")

            perf_metrics = {}
            if device.type == "cuda":
                perf_metrics["cuda_mem_alloc_mb"] = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
                perf_metrics["cuda_mem_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024 ** 2))

            core_metrics = {
                "task_loss": float(fast_ce.item()),
                "batch_supervised_tokens": float(batch_supervised_tokens),
                "batch_supervised_fraction": float(batch_supervised_tokens / max(1, target_ids.numel())),
                "old_task_loss": float(old_loss),
                "forgetting_index": float(forgetting_signal),
                "sleep_pressure": float(scheduler.current_pressure),
                "sleep_count": float(sleep_count),
                "sleep_steps": float(sleep_metrics["steps"]),
                "expert_util_entropy": float(expert_util_entropy),
                "refresh_count": float(refresh_metrics["refresh_count"]),
                "base_teacher_kl": float(base_teacher_kl),
                "distill_kl": float(sleep_metrics["distill_kl"]),
                "ewc_penalty": float(sleep_metrics["ewc_penalty"]),
                "replay_size": float(len(replay)),
                "fast_update_applied": float(fast_update_applied),
                "replay_added_this_step": float(replay_added_this_step),
                "pseudo_batch_size": float(sleep_metrics["pseudo_batch_size"]),
                "replay_batch_size_sleep": float(sleep_metrics["replay_batch_size_sleep"]),
                "sleep_mix_ratio_pseudo": float(sleep_metrics["sleep_mix_ratio_pseudo"]),
                "fisher_replay_samples": float(sleep_metrics["fisher_replay_samples"]),
                "fisher_pseudo_samples": float(sleep_metrics["fisher_pseudo_samples"]),
                **perf_metrics,
            }
            cl_metrics.observe_step_metrics(step, core_metrics)
            core_metrics["non_finite_step_count"] = float(cl_metrics.non_finite_step_count)
            core_metrics["first_non_finite_step"] = float(
                cl_metrics.first_non_finite_step if cl_metrics.first_non_finite_step is not None else -1
            )
            core_metrics["fisher_source_mode"] = str(sleep_metrics["fisher_source_mode"])
            logger.log(core_metrics, step=step)

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

            previous_task_id = current_task

        if previous_task_id is not None:
            boundary_eval = benchmark.evaluate_task_set(model, device, sorted(seen_tasks))
            boundary_metrics = cl_metrics.record_boundary(
                step=cfg.train.max_steps,
                completed_task_id=previous_task_id,
                task_results=boundary_eval,
            )
            boundary_payload = {
                "seen_acc_avg": float(boundary_metrics["seen_acc_avg"]),
                "seen_loss_avg": float(boundary_metrics["seen_loss_avg"]),
                "forgetting": float(boundary_metrics["forgetting"]),
                "bwt": float(boundary_metrics["bwt"]),
                "tasks_completed": float(boundary_metrics["tasks_completed"]),
                "boundary_index": float(boundary_metrics["boundary_index"]),
                "boundary_step": float(boundary_metrics["boundary_step"]),
            }
            for task_id, result in boundary_eval.items():
                boundary_payload[f"task_acc/{task_id}"] = float(result.token_acc)
                boundary_payload[f"task_loss/{task_id}"] = float(result.loss)
                boundary_payload[f"task_tokens/{task_id}"] = float(result.n_tokens)
            logger.log(boundary_payload, step=cfg.train.max_steps)

        cl_summary = cl_metrics.summary()
        final_summary = {
            "sleep_count": float(sleep_count),
            "refresh_count_total": float(refresh_count_total),
            "base_updates_in_sleep": float(sleep_base_updates),
            "final_forgetting_index": float(max(0.0, old_loss - best_old_loss) if best_old_loss < float("inf") else 0.0),
            "final_old_task_loss": float(old_loss),
            "final_seen_acc_avg": float(cl_summary["final_seen_acc_avg"]),
            "final_forgetting": float(cl_summary["final_forgetting"]),
            "final_bwt": float(cl_summary["final_bwt"]),
            "seen_acc_auc": float(cl_summary["seen_acc_auc"]),
            "tasks_completed": float(cl_summary["tasks_completed"]),
            "non_finite_step_count": float(cl_summary["non_finite_step_count"]),
            "first_non_finite_step": float(cl_summary["first_non_finite_step"]),
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
