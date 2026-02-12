from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDoc:
    metric: str
    scope: str
    direction: str
    unit: str
    definition: str
    formula: str = ""


def _rows(docs: list[MetricDoc]) -> list[dict[str, str]]:
    return [
        {
            "metric": d.metric,
            "scope": d.scope,
            "direction": d.direction,
            "unit": d.unit,
            "definition": d.definition,
            "formula": d.formula,
        }
        for d in docs
    ]


def render_metric_glossary_markdown(title: str, rows: list[dict[str, str]]) -> str:
    lines = [f"# {title}", "", "| Metric | Scope | Direction | Unit | Definition | Formula |", "|---|---|---|---|---|---|"]
    for row in rows:
        vals = [row.get("metric", ""), row.get("scope", ""), row.get("direction", ""), row.get("unit", ""), row.get("definition", ""), row.get("formula", "")]
        vals = [v.replace("|", "\\|") for v in vals]
        lines.append(f"| {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} |")
    lines.append("")
    return "\n".join(lines)


def training_metric_glossary_rows() -> list[dict[str, str]]:
    docs = [
        MetricDoc("step", "global", "n/a", "step", "Global logging step."),
        MetricDoc("task_loss", "wake-step", "lower", "cross-entropy", "Fast-system next-token prediction loss on current batch."),
        MetricDoc("batch_supervised_tokens", "wake-step", "higher", "count", "Number of non-ignored target tokens (`target != -100`) in current batch."),
        MetricDoc("batch_supervised_fraction", "wake-step", "higher", "ratio", "Fraction of supervised tokens in current batch: supervised_tokens / total_tokens."),
        MetricDoc("old_task_loss", "wake-step", "lower", "cross-entropy", "Mean loss on held-out old tasks during periodic eval."),
        MetricDoc("forgetting_index", "wake-step", "lower", "loss delta", "Current old-task-loss minus best observed old-task-loss so far.", "max(0, old_task_loss - best_old_task_loss)"),
        MetricDoc("sleep_pressure", "wake-step", "lower", "score", "Scheduler pressure combining saturation, forgetting, and novelty."),
        MetricDoc("sleep_count", "wake-step+summary", "n/a", "count", "Number of sleep cycles triggered so far."),
        MetricDoc("sleep_steps", "wake-step", "n/a", "steps", "Base-model consolidation steps executed in most recent sleep cycle."),
        MetricDoc("expert_util_entropy", "wake-step", "higher", "nats", "Entropy of expert activation counts; higher means more even expert usage."),
        MetricDoc("refresh_count", "wake-step", "n/a", "count", "Experts refreshed in the most recent refresh stage."),
        MetricDoc("base_teacher_kl", "wake-step", "lower", "KL", "KL(base-only || teacher(base+experts)) on current sample."),
        MetricDoc("distill_kl", "wake-step", "lower", "KL", "Mean KL distillation term during latest sleep cycle."),
        MetricDoc("ewc_penalty", "wake-step", "n/a", "penalty", "EWC regularization penalty applied during sleep updates."),
        MetricDoc("replay_size", "wake-step", "n/a", "count", "Current replay buffer size."),
        MetricDoc("replay_added_this_step", "wake-step", "n/a", "count", "Episodes added to replay at this wake step; expected to match batch size."),
        MetricDoc("fast_update_applied", "wake-step", "n/a", "flag", "1 when hippocampal fast optimizer updated this step, else 0."),
        MetricDoc("pseudo_batch_size", "wake-step", "n/a", "count", "Pseudo-rehearsal examples mixed into the latest sleep distillation batch."),
        MetricDoc("replay_batch_size_sleep", "wake-step", "n/a", "count", "Replay examples used in the latest sleep distillation batch."),
        MetricDoc("sleep_mix_ratio_pseudo", "wake-step", "n/a", "ratio", "Pseudo fraction in latest sleep distillation mix: pseudo / (replay + pseudo)."),
        MetricDoc("fisher_replay_samples", "wake-step", "n/a", "count", "Replay samples used to update Fisher during latest sleep cycle."),
        MetricDoc("fisher_pseudo_samples", "wake-step", "n/a", "count", "Pseudo samples used to update Fisher during latest sleep cycle."),
        MetricDoc("fisher_source_mode", "wake-step", "n/a", "category", "Fisher source mode: mixed_replay_pseudo or replay_only."),
        MetricDoc("pseudo_source_mode", "wake-step", "n/a", "category", "Pseudo data source mode: mixed, replay_echo, or random_tokens."),
        MetricDoc("hf_injection_mode", "runtime", "n/a", "category", "HF injection mode: decoder_split or legacy_head_only."),
        MetricDoc("hf_injection_layer_idx", "runtime", "n/a", "index", "HF decoder layer index where injection is applied (decoder_split mode)."),
        MetricDoc("hf_num_decoder_layers", "runtime", "n/a", "count", "Total decoder layers resolved for HF split injection."),
        MetricDoc("hf_injection_fraction", "runtime", "n/a", "fraction", "Configured fraction used to compute HF split depth."),
        MetricDoc("hf_injection_gc_policy", "runtime", "n/a", "category", "Policy for gradient checkpointing during split injection."),
        MetricDoc("core/*", "wandb-history", "mixed", "mixed", "Curated W&B dashboard metrics namespace used to keep run dashboards compact."),
        MetricDoc("boundary/*", "wandb-history", "mixed", "mixed", "Curated task-boundary W&B dashboard metrics namespace."),
        MetricDoc("cuda_mem_alloc_mb", "wake-step", "lower", "MB", "CUDA allocated memory snapshot."),
        MetricDoc("cuda_mem_reserved_mb", "wake-step", "lower", "MB", "CUDA reserved memory snapshot."),
        MetricDoc("non_finite_step_count", "wake-step+summary", "lower", "count", "Number of training steps where tracked scalars were NaN/Inf."),
        MetricDoc("first_non_finite_step", "wake-step+summary", "n/a", "step", "First step with non-finite tracked scalar, or -1."),
        MetricDoc("seen_acc_avg", "boundary", "higher", "accuracy", "Mean token accuracy over all seen tasks at this boundary.", "mean(A[i][j] for j<=i)"),
        MetricDoc("seen_loss_avg", "boundary", "lower", "loss", "Mean token loss over all seen tasks at this boundary.", "mean(L[i][j] for j<=i)"),
        MetricDoc("forgetting", "boundary+summary", "lower", "accuracy drop", "Mean drop from each old task's historical best accuracy to current boundary.", "mean(max_{k<=i-1} A[k][j] - A[i][j] for j<i)"),
        MetricDoc("bwt", "boundary+summary", "higher", "accuracy delta", "Backward transfer relative to each task's own post-training boundary.", "mean(A[i][j] - A[j][j] for j<i)"),
        MetricDoc("tasks_completed", "boundary+summary", "higher", "count", "Number of tasks seen/completed at the boundary."),
        MetricDoc("boundary_index", "boundary", "n/a", "index", "0-based boundary index in continual task order."),
        MetricDoc("boundary_step", "boundary", "n/a", "step", "Training step at which boundary evaluation was logged."),
        MetricDoc("task_acc/{task_id}", "boundary", "higher", "accuracy", "Per-task token accuracy for task_id at this boundary."),
        MetricDoc("task_loss/{task_id}", "boundary", "lower", "loss", "Per-task token loss for task_id at this boundary."),
        MetricDoc("task_tokens/{task_id}", "boundary", "n/a", "count", "Evaluated token count for task_id at this boundary."),
        MetricDoc("refresh_count_total", "summary", "n/a", "count", "Total experts refreshed over whole run."),
        MetricDoc("base_updates_in_sleep", "summary", "n/a", "count", "Total base-model optimizer updates performed during sleep."),
        MetricDoc("final_forgetting_index", "summary", "lower", "loss delta", "Final step-level forgetting index."),
        MetricDoc("final_old_task_loss", "summary", "lower", "loss", "Final periodic old-task loss."),
        MetricDoc("final_seen_acc_avg", "summary", "higher", "accuracy", "Final boundary seen-task average accuracy."),
        MetricDoc("final_forgetting", "summary", "lower", "accuracy drop", "Final boundary forgetting score."),
        MetricDoc("final_bwt", "summary", "higher", "accuracy delta", "Final boundary backward transfer score."),
        MetricDoc("seen_acc_auc", "summary", "higher", "AUC", "Boundary-trajectory AUC of seen-task average accuracy."),
    ]
    return _rows(docs)


def continual_dashboard_metric_glossary_rows() -> list[dict[str, str]]:
    docs = [
        MetricDoc("final_seen_acc_avg", "suite_summary.csv", "higher", "accuracy", "Final boundary seen-task average token accuracy."),
        MetricDoc("final_forgetting", "suite_summary.csv", "lower", "accuracy drop", "Final boundary forgetting score."),
        MetricDoc("final_bwt", "suite_summary.csv", "higher", "accuracy delta", "Final boundary backward transfer."),
        MetricDoc("seen_acc_auc", "suite_summary.csv", "higher", "AUC", "AUC over seen-task average accuracy trajectory."),
        MetricDoc("seen_acc_avg", "boundary_metrics.csv", "higher", "accuracy", "Boundary seen-task average token accuracy."),
        MetricDoc("seen_loss_avg", "boundary_metrics.csv", "lower", "loss", "Boundary seen-task average token loss."),
        MetricDoc("forgetting", "boundary_metrics.csv", "lower", "accuracy drop", "Boundary forgetting score."),
        MetricDoc("bwt", "boundary_metrics.csv", "higher", "accuracy delta", "Boundary backward transfer score."),
        MetricDoc("token_acc", "task_matrix.csv", "higher", "accuracy", "Per-task boundary token accuracy."),
        MetricDoc("loss", "task_matrix.csv", "lower", "loss", "Per-task boundary token loss."),
        MetricDoc("n_tokens", "task_matrix.csv", "n/a", "count", "Token count used for per-task boundary eval."),
        MetricDoc("delta_final_seen_acc_avg", "pair_table", "higher", "accuracy delta", "Baseline A minus baseline B for final seen accuracy."),
        MetricDoc("delta_final_forgetting", "pair_table", "lower", "accuracy drop delta", "Baseline A minus baseline B for final forgetting."),
        MetricDoc("delta_final_bwt", "pair_table", "higher", "accuracy delta", "Baseline A minus baseline B for final BWT."),
        MetricDoc("delta_seen_acc_auc", "pair_table", "higher", "AUC delta", "Baseline A minus baseline B for seen accuracy AUC."),
    ]
    return _rows(docs)


def toy_dashboard_metric_glossary_rows() -> list[dict[str, str]]:
    docs = [
        MetricDoc("final_forgetting", "runs_table", "lower", "loss delta", "Final step-level forgetting index for a run."),
        MetricDoc("mean_forgetting", "runs_table", "lower", "loss delta", "Mean forgetting index across run history."),
        MetricDoc("final_old_task_loss", "runs_table", "lower", "loss", "Final old-task loss."),
        MetricDoc("mean_old_task_loss", "runs_table", "lower", "loss", "Mean old-task loss across run history."),
        MetricDoc("delta_final_forgetting_nosleep_minus_sleep", "pairs_table", "higher", "delta", "No-sleep minus sleep final forgetting; positive favors sleep."),
        MetricDoc("delta_mean_forgetting_nosleep_minus_sleep", "pairs_table", "higher", "delta", "No-sleep minus sleep mean forgetting; positive favors sleep."),
        MetricDoc("delta_final_old_task_loss_nosleep_minus_sleep", "pairs_table", "higher", "delta", "No-sleep minus sleep final old-task loss; positive favors sleep."),
    ]
    return _rows(docs)
