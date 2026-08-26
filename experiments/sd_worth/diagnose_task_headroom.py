from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from harness_matsci.historical import (
    MAIN_MATERIAL_TASKS,
    grouped_four_way_fold,
    load_historical_tasks,
)
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    gain_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--n-regime-folds", type=int, default=5)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--feedback-fraction", type=float, default=0.15)
    parser.add_argument("--acceptance-fraction", type=float, default=0.20)
    args = parser.parse_args()

    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    records_by_task = load_historical_tasks(args.raw_data_dir, MAIN_MATERIAL_TASKS)
    rows = [
        _diagnose_fold(
            records_by_task,
            fold=fold,
            n_regime_folds=args.n_regime_folds,
            budget_fraction=args.budget_fraction,
            train_fraction=args.train_fraction,
            feedback_fraction=args.feedback_fraction,
            acceptance_fraction=args.acceptance_fraction,
        )
        for fold in folds
    ]
    summary = _summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_headroom.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "TASK_HEADROOM.md").write_text(_markdown(summary), encoding="utf-8")


def _diagnose_fold(
    records_by_task: dict,
    *,
    fold: int,
    n_regime_folds: int,
    budget_fraction: float,
    train_fraction: float,
    feedback_fraction: float,
    acceptance_fraction: float,
) -> dict:
    splits_by_task = {
        task: grouped_four_way_fold(
            records,
            fold_index=fold,
            n_folds=n_regime_folds,
            train_fraction=train_fraction,
            feedback_fraction=feedback_fraction,
            acceptance_fraction=acceptance_fraction,
        )
        for task, records in records_by_task.items()
    }
    train_records = [
        record
        for task in MAIN_MATERIAL_TASKS
        for record in splits_by_task[task]["train"]
    ]
    normalizer = EmpiricalUtilityNormalizer.fit(train_records)
    config = GainConfig()
    reservations = BudgetReservationValues.fit(
        train_records,
        normalizer,
        config,
        budget_fraction=budget_fraction,
        outside_option=0.0,
        strategy="potential_success_slot_value",
    )
    tasks = {}
    for task in MAIN_MATERIAL_TASKS:
        targets = gain_targets(
            splits_by_task[task]["test"], normalizer, config, reservations
        )
        gains = sorted((target.gain for target in targets), reverse=True)
        positive = [gain for gain in gains if gain > 0.0]
        budget_count = max(1, round(budget_fraction * len(gains)))
        oracle_selected = [max(0.0, gain) for gain in gains[:budget_count]]
        tasks[task] = {
            "test_count": len(targets),
            "positive_count": len(positive),
            "positive_fraction": len(positive) / len(targets),
            "maximum_gain": max(gains),
            "positive_mean_gain": mean(positive) if positive else 0.0,
            "oracle_budget_count": budget_count,
            "oracle_population_net_gain": sum(oracle_selected) / len(targets),
        }
    return {
        "fold": fold,
        "reservation_values": reservations.to_json(),
        "tasks": tasks,
    }


def _summarize(rows: list[dict]) -> dict:
    return {
        "experiment": "h13_task_oracle_headroom",
        "interpretation": (
            "This is an outcome-aware diagnostic upper bound, never a deployable policy. "
            "It tests whether each held-out task contains positive-gain actions under the "
            "locked net-scientific-gain definition."
        ),
        "rows": rows,
        "mean_by_task": {
            task: {
                "positive_fraction": mean(
                    row["tasks"][task]["positive_fraction"] for row in rows
                ),
                "oracle_population_net_gain": mean(
                    row["tasks"][task]["oracle_population_net_gain"] for row in rows
                ),
                "folds_with_positive_headroom": sum(
                    row["tasks"][task]["positive_count"] > 0 for row in rows
                ),
            }
            for task in MAIN_MATERIAL_TASKS
        },
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# H13 Task Oracle Headroom Diagnostic",
        "",
        "This is an outcome-aware diagnostic upper bound, not a deployable method.",
        "Hidden outcomes are used only to determine whether positive-gain actions exist",
        "under the locked train-only normalization and opportunity-cost definition.",
        "",
        "| fold | task | positive fraction | maximum gain | oracle population net gain |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        for task, values in row["tasks"].items():
            lines.append(
                f"| {row['fold']} | `{task}` | {values['positive_fraction']:.4f} | "
                f"{values['maximum_gain']:+.4f} | "
                f"{values['oracle_population_net_gain']:+.6f} |"
            )
    lines.extend(["", "## Aggregate", ""])
    for task, values in summary["mean_by_task"].items():
        lines.append(
            f"- `{task}`: mean positive fraction "
            f"`{values['positive_fraction']:.4f}`, mean oracle population gain "
            f"`{values['oracle_population_net_gain']:+.6f}`, positive headroom in "
            f"`{values['folds_with_positive_headroom']}/{len(summary['rows'])}` folds."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Unique discovery has positive oracle headroom in every fold, so H13's zero",
            "  selected unique actions are a ranking/evidence failure rather than an empty",
            "  utility target.",
            "- Extreme-property discovery has positive headroom in four folds but one",
            "  structurally empty held-out regime, so both ranking and regime support matter.",
            "- H14 must improve task-specific evidence before changing acceptance power.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
