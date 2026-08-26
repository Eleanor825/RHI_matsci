from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


PRIMARY = "task_sparse_tool_margin_continuous_sd_worth"
NO_SOURCE = "task_sparse_tool_margin_continuous_mean"
BASELINES = (
    "base_mean",
    "base_ensemble_lcb",
    "base_verbal_confidence",
    "base_evidence_heuristic",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    paths = sorted(results_dir.glob("fold_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
    if not paths:
        raise ValueError(f"no fold results in {results_dir}")
    rows = [_row(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    summary = _summary(rows)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (results_dir / "RESULTS.md").write_text(_markdown(summary), encoding="utf-8")


def _row(result: dict) -> dict:
    primary = result["safe_activation"]["source_decomposed"]
    no_source = result["safe_activation"]["no_source"]
    baselines = {
        method: result["methods"][method]["risk_control"]["alpha_0.20"]["test"]
        for method in BASELINES
    }
    primary_test = primary["test"]
    no_source_test = no_source["test"]
    return {
        "fold": int(result["regime_fold"]),
        "audit": bool(result["nontraining_outcome_invariance_audit"]["passed"]),
        "policy": primary["selected_policy"],
        "coverage": float(primary_test["coverage"]),
        "risk": float(primary_test["selective_risk"]),
        "selected_gain": float(primary_test["selected_mean_gain"]),
        "population_net_gain": float(primary_test["population_net_gain"]),
        "tasks_selected": sorted(primary_test["task_risks"]),
        "no_source_gain": float(no_source_test["population_net_gain"]),
        "delta_no_source": float(primary_test["population_net_gain"])
        - float(no_source_test["population_net_gain"]),
        "baseline_gains": {
            method: float(values["population_net_gain"])
            for method, values in baselines.items()
        },
        "test_groups_by_task": result["test_groups_by_task"],
    }


def _summary(rows: list[dict]) -> dict:
    tasks_selected = sorted({task for row in rows for task in row["tasks_selected"]})
    mean_baseline_gains = {
        method: mean(row["baseline_gains"][method] for row in rows)
        for method in BASELINES
    }
    return {
        "experiment": "h13_powered_acceptance_confirmatory",
        "rows": rows,
        "folds": len(rows),
        "audit_pass_folds": sum(row["audit"] for row in rows),
        "active_folds": sum(row["policy"] != "no_op" for row in rows),
        "nonzero_coverage_folds": sum(row["coverage"] > 0.0 for row in rows),
        "positive_gain_folds": sum(row["population_net_gain"] > 0.0 for row in rows),
        "risk_at_most_0.20_folds": sum(row["risk"] <= 0.20 for row in rows),
        "mean_coverage": mean(row["coverage"] for row in rows),
        "mean_risk": mean(row["risk"] for row in rows),
        "mean_population_net_gain": mean(row["population_net_gain"] for row in rows),
        "mean_no_source_gain": mean(row["no_source_gain"] for row in rows),
        "mean_delta_no_source": mean(row["delta_no_source"] for row in rows),
        "source_wins": sum(row["delta_no_source"] > 0.0 for row in rows),
        "tasks_selected": tasks_selected,
        "mean_baseline_gains": mean_baseline_gains,
        "confirmatory_pass": (
            sum(row["population_net_gain"] > 0.0 and row["coverage"] > 0.0 for row in rows)
            >= 3
            and all(row["risk"] <= 0.20 for row in rows if row["policy"] != "no_op")
            and mean(row["delta_no_source"] for row in rows) > 0.0
            and len(tasks_selected) >= 2
            and all(row["audit"] for row in rows)
        ),
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# H13 Powered-Acceptance Confirmatory Results",
        "",
        "Fold 0 is excluded. These rows are untouched regime folds 1--4.",
        "",
        "| fold | policy | coverage | risk | selected gain | population net gain | delta vs no-source | tasks |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['fold']} | `{row['policy']}` | {row['coverage']:.4f} | "
            f"{row['risk']:.4f} | {row['selected_gain']:.4f} | "
            f"{row['population_net_gain']:+.6f} | {row['delta_no_source']:+.6f} | "
            f"`{','.join(row['tasks_selected'])}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Active folds: `{summary['active_folds']}/{summary['folds']}`.",
            f"- Positive-gain folds: `{summary['positive_gain_folds']}/{summary['folds']}`.",
            f"- Risk at most 0.20: `{summary['risk_at_most_0.20_folds']}/{summary['folds']}`.",
            f"- Mean coverage: `{summary['mean_coverage']:.6f}`.",
            f"- Mean population net gain: `{summary['mean_population_net_gain']:+.6f}`.",
            f"- Mean no-source gain: `{summary['mean_no_source_gain']:+.6f}`.",
            f"- Mean source advantage: `{summary['mean_delta_no_source']:+.6f}`; wins `{summary['source_wins']}/{summary['folds']}`.",
            f"- Tasks selected: `{','.join(summary['tasks_selected'])}`.",
            f"- Locked confirmatory pass: `{summary['confirmatory_pass']}`.",
            "",
            "## Static baselines",
            "",
        ]
    )
    for method, value in summary["mean_baseline_gains"].items():
        lines.append(f"- `{method}` mean population net gain: `{value:+.6f}`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
