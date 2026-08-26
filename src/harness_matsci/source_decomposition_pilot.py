from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from .features import feature_names_from_records
from .historical import MAIN_MATERIAL_TASKS, load_historical_tasks, split_historical_tasks
from .replica_contract import (
    ActionReplicaGrid,
    AgentEvidencePrediction,
    controlled_evidence_views,
)
from .scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    fit_gain_distribution,
    gain_targets,
)
from .sd_worth_experiment import _with_task_features
from .source_decomposition import decompose_source_variance


AGENT_BOOTSTRAP_SEEDS = (101, 202, 303)


def run_source_decomposition_pilot(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    split_seed: int = 7,
    budget_fraction: float = 0.10,
    permutation_repeats: int = 40,
) -> dict[str, Any]:
    records_by_task = load_historical_tasks(data_dir, MAIN_MATERIAL_TASKS)
    task_splits = split_historical_tasks(records_by_task, seed=split_seed)
    splits = {
        split: _with_task_features(
            [record for task in MAIN_MATERIAL_TASKS for record in task_splits[task][split]]
        )
        for split in ("train", "feedback", "test")
    }
    config = GainConfig()
    normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
    reservation = BudgetReservationValues.fit(
        splits["train"], normalizer, config, budget_fraction=budget_fraction
    )
    targets = {
        split: gain_targets(records, normalizer, config, reservation)
        for split, records in splits.items()
    }
    feature_names = feature_names_from_records(splits["train"])
    models = []
    for seed in AGENT_BOOTSTRAP_SEEDS:
        rng = random.Random(seed)
        sampled = [rng.randrange(len(splits["train"])) for _ in splits["train"]]
        models.append(
            (
                f"bootstrap-gain::{seed}",
                fit_gain_distribution(
                    [splits["train"][index] for index in sampled],
                    [targets["train"][index] for index in sampled],
                    splits["feedback"],
                    targets["feedback"],
                    feature_names=feature_names,
                    l2=0.1,
                ),
            )
        )

    rows = []
    for record, target in zip(splits["test"], targets["test"]):
        views = controlled_evidence_views(record)
        predictions = []
        view_means: dict[str, list[float]] = {view.view_id: [] for view in views}
        for agent_id, model in models:
            model_predictions = model.predict([view.record for view in views])
            for view, prediction in zip(views, model_predictions):
                view_means[view.view_id].append(prediction.mean_gain)
                predictions.append(
                    AgentEvidencePrediction(
                        record_id=record.record_id,
                        agent_replica_id=agent_id,
                        evidence_view_id=view.view_id,
                        predicted_gain=prediction.mean_gain,
                        positive_gain_probability=prediction.action_worthiness,
                        backend="bootstrap_gain_head",
                        internal_signals={"gain_scale": prediction.gain_scale},
                        external_signals={
                            key: float(view.record.features.get(key, 0.0))
                            for key in (
                                "evidence_support",
                                "evidence_conflict",
                                "source_reliability",
                                "tool_agreement",
                                "ood_score",
                                "source_risk",
                            )
                        },
                    )
                )
        grid = ActionReplicaGrid.from_predictions(predictions)
        summary = grid.summarize()
        full_mean = mean(view_means["evidence::full"])
        dropout_mean = mean(view_means["evidence::dropout"])
        conflict_mean = mean(view_means["evidence::conflict"])
        permuted_external = _permuted_external_variance(
            grid.predicted_gain_matrix(),
            repeats=permutation_repeats,
            seed=_stable_seed(record.record_id),
        )
        rows.append(
            {
                "record_id": record.record_id,
                "benchmark": record.benchmark,
                "group_id": str(record.metadata.get("group_id", record.benchmark)),
                "target_gain": target.gain,
                "worthy": target.worthy,
                **summary.to_json(),
                "full_view_mean_gain": full_mean,
                "dropout_shift": dropout_mean - full_mean,
                "conflict_shift": conflict_mean - full_mean,
                "permuted_external_variance_mean": mean(permuted_external),
                "external_exceeds_permutation_fraction": mean(
                    float(summary.source_variance.external_variance > value)
                    for value in permuted_external
                ),
            }
        )

    report = {
        "schema_version": "sd-worth-h2-pilot-v1",
        "protocol": "research/evl_iclr/experiments/h2_variance_decomposition/protocol_pilot.md",
        "split_seed": split_seed,
        "agent_bootstrap_seeds": list(AGENT_BOOTSTRAP_SEEDS),
        "evidence_views": ["full", "dropout", "conflict"],
        "n_actions": len(rows),
        "summary": _summarize(rows),
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "action_summaries.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (destination / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (destination / "RESULTS.md").write_text(_markdown(report))
    return report


def _permuted_external_variance(
    matrix: list[list[float]], *, repeats: int, seed: int
) -> list[float]:
    rng = random.Random(seed)
    output = []
    for _ in range(repeats):
        permuted = []
        for row in matrix:
            shuffled = list(row)
            rng.shuffle(shuffled)
            permuted.append(shuffled)
        output.append(decompose_source_variance(permuted).external_variance)
    return output


def _stable_seed(value: str) -> int:
    output = 0
    for character in value:
        output = (output * 131 + ord(character)) % (2**32)
    return output


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _summarize_group(rows),
        "by_task": {
            task: _summarize_group([row for row in rows if row["benchmark"] == task])
            for task in MAIN_MATERIAL_TASKS
        },
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0}
    internal = [row["source_variance"]["internal_variance"] for row in rows]
    external = [row["source_variance"]["external_variance"] for row in rows]
    total = [row["source_variance"]["total_variance"] for row in rows]
    errors = [abs(row["full_view_mean_gain"] - row["target_gain"]) for row in rows]
    return {
        "n": len(rows),
        "mean_internal_variance": mean(internal),
        "mean_external_variance": mean(external),
        "mean_total_variance": mean(total),
        "mean_dropout_shift": mean(row["dropout_shift"] for row in rows),
        "mean_absolute_dropout_shift": mean(abs(row["dropout_shift"]) for row in rows),
        "mean_conflict_shift": mean(row["conflict_shift"] for row in rows),
        "mean_absolute_conflict_shift": mean(abs(row["conflict_shift"]) for row in rows),
        "external_exceeds_permutation_fraction": mean(
            row["external_exceeds_permutation_fraction"] for row in rows
        ),
        "external_variance_error_correlation": _correlation(external, errors),
        "internal_variance_error_correlation": _correlation(internal, errors),
    }


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((a - left_mean) ** 2 for a in left) ** 0.5
    right_scale = sum((b - right_mean) ** 2 for b in right) ** 0.5
    return numerator / (left_scale * right_scale) if left_scale > 0 and right_scale > 0 else 0.0


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# H2 Source-Decomposition Pilot",
        "",
        "This is a mechanism check using bootstrap gain heads and controlled evidence interventions; it is not the final LLM/MatBot experiment.",
        "",
        "| Scope | N | Internal var | External var | |Dropout shift| | |Conflict shift| | External > permuted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {"overall": report["summary"]["overall"], **report["summary"]["by_task"]}
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['n']} | {values.get('mean_internal_variance', 0):.6f} | "
            f"{values.get('mean_external_variance', 0):.6f} | "
            f"{values.get('mean_absolute_dropout_shift', 0):.6f} | "
            f"{values.get('mean_absolute_conflict_shift', 0):.6f} | "
            f"{values.get('external_exceeds_permutation_fraction', 0):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the balanced source-decomposition pilot.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--permutation-repeats", type=int, default=40)
    args = parser.parse_args()
    report = run_source_decomposition_pilot(
        args.data_dir,
        args.out_dir,
        split_seed=args.split_seed,
        budget_fraction=args.budget_fraction,
        permutation_repeats=args.permutation_repeats,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
