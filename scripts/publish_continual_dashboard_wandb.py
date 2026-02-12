from __future__ import annotations

import argparse
import csv
import re
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb


@dataclass
class RunSummary:
    run_id: str
    run_name: str
    baseline: str
    seed: int
    final_seen_acc_avg: float
    final_forgetting: float
    final_bwt: float
    seen_acc_auc: float


def _parse_baseline_seed(run_name: str) -> tuple[str, int] | None:
    m = re.match(r"(.+)-seed(\d+)$", run_name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _aggregate_boundary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for row in rows:
        key = (str(row["baseline"]), int(row["boundary_index"]))
        for metric in ("seen_acc_avg", "seen_loss_avg", "forgetting", "bwt"):
            buckets[key][metric].append(float(row[metric]))

    out: list[dict[str, Any]] = []
    for (baseline, boundary_index), vals in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        rec: dict[str, Any] = {
            "baseline": baseline,
            "boundary_index": boundary_index,
        }
        for metric in ("seen_acc_avg", "seen_loss_avg", "forgetting", "bwt"):
            rec[f"mean_{metric}"] = stats.mean(vals[metric]) if vals[metric] else 0.0
        out.append(rec)
    return out


def _collect(
    *,
    entity: str,
    project: str,
    group: str,
) -> tuple[list[RunSummary], list[dict[str, Any]], list[dict[str, Any]]]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    summaries: list[RunSummary] = []
    boundary_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for run in runs:
        run_name = (run.name or "").strip()
        parsed = _parse_baseline_seed(run_name)
        if parsed is None:
            continue
        baseline, seed = parsed

        summary = run.summary or {}
        summaries.append(
            RunSummary(
                run_id=run.id,
                run_name=run_name,
                baseline=baseline,
                seed=seed,
                final_seen_acc_avg=_as_float(summary.get("final_seen_acc_avg", 0.0)),
                final_forgetting=_as_float(summary.get("final_forgetting", 0.0)),
                final_bwt=_as_float(summary.get("final_bwt", 0.0)),
                seen_acc_auc=_as_float(summary.get("seen_acc_auc", 0.0)),
            )
        )

        for row in run.scan_history():
            if row.get("boundary_index") is None:
                continue
            boundary_index = int(_as_float(row.get("boundary_index", 0.0)))
            boundary_rows.append(
                {
                    "run_id": run.id,
                    "run_name": run_name,
                    "baseline": baseline,
                    "seed": seed,
                    "step": int(_as_float(row.get("step", 0.0))),
                    "boundary_index": boundary_index,
                    "boundary_step": int(_as_float(row.get("boundary_step", 0.0))),
                    "seen_acc_avg": _as_float(row.get("seen_acc_avg", 0.0)),
                    "seen_loss_avg": _as_float(row.get("seen_loss_avg", 0.0)),
                    "forgetting": _as_float(row.get("forgetting", 0.0)),
                    "bwt": _as_float(row.get("bwt", 0.0)),
                    "tasks_completed": _as_float(row.get("tasks_completed", 0.0)),
                }
            )

            for key, value in row.items():
                if not key.startswith("task_acc/"):
                    continue
                task_id = key.split("/", 1)[1]
                task_rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run_name,
                        "baseline": baseline,
                        "seed": seed,
                        "boundary_index": boundary_index,
                        "task_id": int(task_id),
                        "token_acc": _as_float(value, 0.0),
                        "loss": _as_float(row.get(f"task_loss/{task_id}", 0.0), 0.0),
                        "n_tokens": int(_as_float(row.get(f"task_tokens/{task_id}", 0.0))),
                    }
                )

    summaries.sort(key=lambda x: (x.baseline, x.seed, x.run_name))
    boundary_rows.sort(key=lambda x: (x["baseline"], x["seed"], x["boundary_index"]))
    task_rows.sort(key=lambda x: (x["baseline"], x["seed"], x["boundary_index"], x["task_id"]))
    return summaries, boundary_rows, task_rows


def _paired_rows(rows: list[RunSummary], baseline_a: str, baseline_b: str) -> list[dict[str, Any]]:
    by_seed_a = {r.seed: r for r in rows if r.baseline == baseline_a}
    by_seed_b = {r.seed: r for r in rows if r.baseline == baseline_b}
    out: list[dict[str, Any]] = []
    for seed in sorted(set(by_seed_a.keys()) & set(by_seed_b.keys())):
        a = by_seed_a[seed]
        b = by_seed_b[seed]
        out.append(
            {
                "seed": seed,
                "baseline_a": baseline_a,
                "baseline_b": baseline_b,
                "delta_final_seen_acc_avg": a.final_seen_acc_avg - b.final_seen_acc_avg,
                "delta_final_forgetting": a.final_forgetting - b.final_forgetting,
                "delta_final_bwt": a.final_bwt - b.final_bwt,
                "delta_seen_acc_auc": a.seen_acc_auc - b.seen_acc_auc,
            }
        )
    return out


def _make_bar_table(rows: list[RunSummary], metric: str) -> wandb.Table:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[row.baseline].append(float(getattr(row, metric)))

    table = wandb.Table(columns=["baseline", "mean", "std", "n"])
    for baseline in sorted(buckets.keys()):
        vals = buckets[baseline]
        mean = stats.mean(vals)
        std = stats.stdev(vals) if len(vals) > 1 else 0.0
        table.add_data(baseline, mean, std, len(vals))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish continual-learning suite dashboard to W&B and CSV.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--baseline-a", default="full_cls")
    parser.add_argument("--baseline-b", default="no_sleep")
    parser.add_argument("--output-root", default="outputs/eval")
    parser.add_argument("--analysis-run-name", default=None)
    args = parser.parse_args()

    summaries, boundary_rows, task_rows = _collect(entity=args.entity, project=args.project, group=args.group)
    if not summaries:
        raise RuntimeError(f"No runs found for group '{args.group}' in {args.entity}/{args.project}.")

    output_dir = Path(args.output_root) / args.group
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows = [
        {
            "run_id": r.run_id,
            "run_name": r.run_name,
            "baseline": r.baseline,
            "seed": r.seed,
            "final_seen_acc_avg": r.final_seen_acc_avg,
            "final_forgetting": r.final_forgetting,
            "final_bwt": r.final_bwt,
            "seen_acc_auc": r.seen_acc_auc,
        }
        for r in summaries
    ]
    _write_csv(
        output_dir / "suite_summary.csv",
        suite_rows,
        [
            "run_id",
            "run_name",
            "baseline",
            "seed",
            "final_seen_acc_avg",
            "final_forgetting",
            "final_bwt",
            "seen_acc_auc",
        ],
    )
    _write_csv(
        output_dir / "boundary_metrics.csv",
        boundary_rows,
        [
            "run_id",
            "run_name",
            "baseline",
            "seed",
            "step",
            "boundary_index",
            "boundary_step",
            "seen_acc_avg",
            "seen_loss_avg",
            "forgetting",
            "bwt",
            "tasks_completed",
        ],
    )
    _write_csv(
        output_dir / "task_matrix.csv",
        task_rows,
        [
            "run_id",
            "run_name",
            "baseline",
            "seed",
            "boundary_index",
            "task_id",
            "token_acc",
            "loss",
            "n_tokens",
        ],
    )

    boundary_agg = _aggregate_boundary(boundary_rows)
    pair_rows = _paired_rows(summaries, args.baseline_a, args.baseline_b)

    analysis_run_name = args.analysis_run_name or f"cl-ni8-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=analysis_run_name,
        group=f"analysis-{args.group}",
        job_type="analysis",
        config={
            "source_group": args.group,
            "baseline_a": args.baseline_a,
            "baseline_b": args.baseline_b,
            "suite_rows": len(suite_rows),
        },
    )

    suite_table = wandb.Table(columns=list(suite_rows[0].keys()))
    for row in suite_rows:
        suite_table.add_data(*[row[c] for c in suite_rows[0].keys()])

    pair_table = wandb.Table(columns=list(pair_rows[0].keys()) if pair_rows else ["seed"])
    for row in pair_rows:
        pair_table.add_data(*[row[c] for c in pair_rows[0].keys()])

    final_seen_tbl = _make_bar_table(summaries, "final_seen_acc_avg")
    final_forgetting_tbl = _make_bar_table(summaries, "final_forgetting")
    final_bwt_tbl = _make_bar_table(summaries, "final_bwt")
    seen_auc_tbl = _make_bar_table(summaries, "seen_acc_auc")

    payload: dict[str, Any] = {
        "continual/suite_table": suite_table,
        "continual/pair_table": pair_table,
        "continual/final_seen_accuracy_table": final_seen_tbl,
        "continual/final_forgetting_table": final_forgetting_tbl,
        "continual/final_bwt_table": final_bwt_tbl,
        "continual/seen_acc_auc_table": seen_auc_tbl,
        "continual/final_seen_accuracy_by_baseline": wandb.plot.bar(
            final_seen_tbl,
            "baseline",
            "mean",
            title="Final Seen Accuracy by Baseline",
        ),
        "continual/final_forgetting_by_baseline": wandb.plot.bar(
            final_forgetting_tbl,
            "baseline",
            "mean",
            title="Final Forgetting by Baseline",
        ),
        "continual/final_bwt_by_baseline": wandb.plot.bar(
            final_bwt_tbl,
            "baseline",
            "mean",
            title="Final BWT by Baseline",
        ),
        "continual/seen_auc_by_baseline": wandb.plot.bar(
            seen_auc_tbl,
            "baseline",
            "mean",
            title="Seen Accuracy AUC by Baseline",
        ),
    }

    if boundary_agg:
        boundary_table = wandb.Table(columns=list(boundary_agg[0].keys()))
        for row in boundary_agg:
            boundary_table.add_data(*[row[c] for c in boundary_agg[0].keys()])
        payload["continual/boundary_agg_table"] = boundary_table

        for metric in ("mean_seen_acc_avg", "mean_forgetting", "mean_bwt"):
            for baseline in sorted({r["baseline"] for r in boundary_agg}):
                b_rows = [r for r in boundary_agg if r["baseline"] == baseline]
                if not b_rows:
                    continue
                t = wandb.Table(columns=["boundary_index", metric])
                for row in b_rows:
                    t.add_data(row["boundary_index"], row[metric])
                payload[f"continual/{baseline}/{metric}_trajectory"] = wandb.plot.line(
                    t,
                    "boundary_index",
                    metric,
                    title=f"{baseline} {metric} trajectory",
                )

    run.log(payload)
    run.summary["source_group"] = args.group
    run.summary["suite_rows"] = len(suite_rows)
    run.summary["pair_rows"] = len(pair_rows)
    run.summary["csv_suite_summary"] = str((output_dir / "suite_summary.csv").as_posix())
    run.summary["csv_boundary_metrics"] = str((output_dir / "boundary_metrics.csv").as_posix())
    run.summary["csv_task_matrix"] = str((output_dir / "task_matrix.csv").as_posix())
    run.finish()

    print(f"Wrote: {output_dir / 'suite_summary.csv'}")
    print(f"Wrote: {output_dir / 'boundary_metrics.csv'}")
    print(f"Wrote: {output_dir / 'task_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
