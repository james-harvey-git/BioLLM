from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import wandb

from biollm_cls.metrics_catalog import render_metric_glossary_markdown, toy_dashboard_metric_glossary_rows


@dataclass
class RunStats:
    run_id: str
    name: str
    mode: str
    seed: int
    steps: int
    final_forgetting: float
    mean_forgetting: float
    final_old_task_loss: float
    mean_old_task_loss: float
    final_task_loss: float
    max_sleep_count: float


def _extract_seed(name: str) -> int | None:
    match = re.search(r"seed(\d+)$", name)
    if match is None:
        return None
    return int(match.group(1))


def _mode_from_name(name: str) -> str | None:
    if name.startswith("sleep-seed"):
        return "sleep"
    if name.startswith("nosleep-seed"):
        return "nosleep"
    return None


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def collect_toy_ablation_runs(entity: str, project: str, group: str | None) -> list[RunStats]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    out: list[RunStats] = []
    for run in runs:
        name = (run.name or "").strip()
        mode = _mode_from_name(name)
        if mode is None:
            continue
        if group and (run.group or "") != group:
            continue

        seed = _extract_seed(name)
        if seed is None:
            continue

        hist = run.history(
            keys=["step", "forgetting_index", "old_task_loss", "task_loss", "sleep_count"],
            pandas=True,
        )
        if hist is None or hist.empty:
            continue

        hist = hist.sort_values("step")
        final = hist.iloc[-1]

        out.append(
            RunStats(
                run_id=run.id,
                name=name,
                mode=mode,
                seed=seed,
                steps=int(final["step"]),
                final_forgetting=_as_float(final.get("forgetting_index", 0.0)),
                mean_forgetting=_as_float(hist["forgetting_index"].mean()),
                final_old_task_loss=_as_float(final.get("old_task_loss", 0.0)),
                mean_old_task_loss=_as_float(hist["old_task_loss"].mean()),
                final_task_loss=_as_float(final.get("task_loss", 0.0)),
                max_sleep_count=_as_float(hist["sleep_count"].max()),
            )
        )

    out.sort(key=lambda r: (r.seed, r.mode))
    return out


def pair_by_seed(rows: list[RunStats]) -> list[dict[str, float | int]]:
    paired: list[dict[str, float | int]] = []
    seeds = sorted({r.seed for r in rows})

    for seed in seeds:
        per_seed = [r for r in rows if r.seed == seed]
        sleep = next((r for r in per_seed if r.mode == "sleep"), None)
        nosleep = next((r for r in per_seed if r.mode == "nosleep"), None)
        if sleep is None or nosleep is None:
            continue

        paired.append(
            {
                "seed": seed,
                "sleep_final_forgetting": sleep.final_forgetting,
                "nosleep_final_forgetting": nosleep.final_forgetting,
                "delta_final_forgetting_nosleep_minus_sleep": nosleep.final_forgetting - sleep.final_forgetting,
                "sleep_mean_forgetting": sleep.mean_forgetting,
                "nosleep_mean_forgetting": nosleep.mean_forgetting,
                "delta_mean_forgetting_nosleep_minus_sleep": nosleep.mean_forgetting - sleep.mean_forgetting,
                "sleep_final_old_task_loss": sleep.final_old_task_loss,
                "nosleep_final_old_task_loss": nosleep.final_old_task_loss,
                "delta_final_old_task_loss_nosleep_minus_sleep": nosleep.final_old_task_loss - sleep.final_old_task_loss,
                "sleep_count": sleep.max_sleep_count,
            }
        )

    return paired


def publish_visuals(
    *,
    entity: str,
    project: str,
    source_group: str | None,
    analysis_run_name: str,
    rows: list[RunStats],
    pairs: list[dict[str, float | int]],
) -> None:
    if not rows:
        raise RuntimeError("No matching toy sleep/nosleep runs found.")

    run = wandb.init(
        entity=entity,
        project=project,
        name=analysis_run_name,
        job_type="analysis",
        group="analysis-toy-ablation",
        config={
            "source_group": source_group,
            "source_runs": [r.name for r in rows],
            "notes": "Toy benchmark only; not instruction-benchmark evidence.",
        },
    )

    runs_columns = [
        "run_id",
        "name",
        "mode",
        "seed",
        "steps",
        "final_forgetting",
        "mean_forgetting",
        "final_old_task_loss",
        "mean_old_task_loss",
        "final_task_loss",
        "max_sleep_count",
    ]
    runs_table = wandb.Table(columns=runs_columns)
    for r in rows:
        runs_table.add_data(
            r.run_id,
            r.name,
            r.mode,
            r.seed,
            r.steps,
            r.final_forgetting,
            r.mean_forgetting,
            r.final_old_task_loss,
            r.mean_old_task_loss,
            r.final_task_loss,
            r.max_sleep_count,
        )

    pair_columns = [
        "seed",
        "sleep_final_forgetting",
        "nosleep_final_forgetting",
        "delta_final_forgetting_nosleep_minus_sleep",
        "sleep_mean_forgetting",
        "nosleep_mean_forgetting",
        "delta_mean_forgetting_nosleep_minus_sleep",
        "sleep_final_old_task_loss",
        "nosleep_final_old_task_loss",
        "delta_final_old_task_loss_nosleep_minus_sleep",
        "sleep_count",
    ]
    pairs_table = wandb.Table(columns=pair_columns)
    for p in pairs:
        pairs_table.add_data(*[p[c] for c in pair_columns])

    payload = {
        "toy_ablation/runs_table": runs_table,
        "toy_ablation/pairs_table": pairs_table,
        "toy_ablation/delta_final_old_task_loss_by_seed": wandb.plot.bar(
            pairs_table,
            "seed",
            "delta_final_old_task_loss_nosleep_minus_sleep",
            title="NoSleep - Sleep (Final Old-Task Loss) by Seed",
        ),
        "toy_ablation/delta_mean_forgetting_by_seed": wandb.plot.bar(
            pairs_table,
            "seed",
            "delta_mean_forgetting_nosleep_minus_sleep",
            title="NoSleep - Sleep (Mean Forgetting) by Seed",
        ),
        "toy_ablation/delta_final_forgetting_by_seed": wandb.plot.bar(
            pairs_table,
            "seed",
            "delta_final_forgetting_nosleep_minus_sleep",
            title="NoSleep - Sleep (Final Forgetting) by Seed",
        ),
    }

    glossary_rows = toy_dashboard_metric_glossary_rows()
    glossary_cols = ["metric", "scope", "direction", "unit", "definition", "formula"]
    glossary_table = wandb.Table(columns=glossary_cols)
    for row in glossary_rows:
        glossary_table.add_data(*[row.get(col, "") for col in glossary_cols])
    payload["toy_ablation/metric_glossary"] = glossary_table

    if pairs:
        delta_old = [float(p["delta_final_old_task_loss_nosleep_minus_sleep"]) for p in pairs]
        delta_mean_forgetting = [float(p["delta_mean_forgetting_nosleep_minus_sleep"]) for p in pairs]
        delta_final_forgetting = [float(p["delta_final_forgetting_nosleep_minus_sleep"]) for p in pairs]

        payload.update(
            {
                "toy_ablation/agg_mean_delta_old_task_loss": stats.mean(delta_old),
                "toy_ablation/agg_mean_delta_mean_forgetting": stats.mean(delta_mean_forgetting),
                "toy_ablation/agg_mean_delta_final_forgetting": stats.mean(delta_final_forgetting),
                "toy_ablation/num_pairs": len(pairs),
            }
        )

    run.log(payload)
    output_dir = Path("outputs/eval/toy_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metric_glossary.json").write_text(json.dumps(glossary_rows, indent=2), encoding="utf-8")
    (output_dir / "metric_glossary.md").write_text(
        render_metric_glossary_markdown("Toy Ablation Dashboard Metrics", glossary_rows),
        encoding="utf-8",
    )
    run.summary["metric_glossary_json"] = str((output_dir / "metric_glossary.json").as_posix())
    run.summary["metric_glossary_md"] = str((output_dir / "metric_glossary.md").as_posix())
    print(f"Published analysis run: {run.url}")
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish toy sleep/no-sleep comparison visuals to W&B.")
    parser.add_argument("--entity", required=True, help="W&B entity slug")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--group", default=None, help="Optional source run group filter")
    parser.add_argument("--analysis-run-name", default=None, help="Name for the generated analysis run")
    args = parser.parse_args()

    analysis_name = args.analysis_run_name or f"toy-ablation-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    rows = collect_toy_ablation_runs(args.entity, args.project, args.group)
    pairs = pair_by_seed(rows)
    publish_visuals(
        entity=args.entity,
        project=args.project,
        source_group=args.group,
        analysis_run_name=analysis_name,
        rows=rows,
        pairs=pairs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
