from __future__ import annotations

import argparse
import json
import math
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
from harness_matsci.sequential_tool_benchmark import RealScientificToolEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--n-regime-folds", type=int, default=5)
    parser.add_argument("--evidence-replicas", default="101,211,307")
    args = parser.parse_args()

    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    evidence_replicas = tuple(
        int(value) for value in args.evidence_replicas.split(",") if value.strip()
    )
    records_by_task = load_historical_tasks(args.raw_data_dir, MAIN_MATERIAL_TASKS)
    rows = [
        _diagnose_fold(
            args.raw_data_dir,
            records_by_task,
            fold=fold,
            n_regime_folds=args.n_regime_folds,
            evidence_replicas=evidence_replicas,
        )
        for fold in folds
    ]
    summary = {"experiment": "real_tool_signal_identifiability", "rows": rows}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tool_signal.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "TOOL_SIGNAL.md").write_text(_markdown(summary), encoding="utf-8")


def _diagnose_fold(
    raw_data_dir: str,
    records_by_task: dict,
    *,
    fold: int,
    n_regime_folds: int,
    evidence_replicas: tuple[int, ...],
) -> dict:
    splits_by_task = {
        task: grouped_four_way_fold(
            records,
            fold_index=fold,
            n_folds=n_regime_folds,
            train_fraction=0.50,
            feedback_fraction=0.15,
            acceptance_fraction=0.20,
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
        budget_fraction=0.10,
        outside_option=0.0,
        strategy="potential_success_slot_value",
    )
    environment = RealScientificToolEnvironment.fit(
        raw_data_dir, train_records, environment_seed=1729 + fold
    )
    tasks = {}
    for task in MAIN_MATERIAL_TASKS:
        records = splits_by_task[task]["test"]
        targets = gain_targets(records, normalizer, config, reservations)
        labels = [target.worthy for target in targets]
        gains = [target.gain for target in targets]
        score_matrix = []
        latent_margins = []
        for record in records:
            scores = []
            latent = None
            for evidence_replica in evidence_replicas:
                observation = environment.observe(
                    record,
                    stage="verification",
                    evidence_replica=evidence_replica,
                    run_seed=2729 + fold,
                )
                scores.append(float(observation.record.features["tool_estimated_margin"]))
                latent = float(observation.latent_margin_for_audit_only)
            score_matrix.append(scores)
            latent_margins.append(float(latent))
        scores = [mean(row) for row in score_matrix]
        external_variances = [_population_variance(row) for row in score_matrix]
        absolute_tool_errors = [
            abs(score - latent) for score, latent in zip(scores, latent_margins)
        ]
        tasks[task] = {
            "count": len(records),
            "worthy_fraction": mean(labels),
            "roc_auc_worthiness": _roc_auc(labels, scores),
            "average_precision_worthiness": _average_precision(labels, scores),
            "gain_score_pearson": _pearson(gains, scores),
            "external_variance_error_pearson": _pearson(
                external_variances, absolute_tool_errors
            ),
            "external_variance_error_spearman": _spearman(
                external_variances, absolute_tool_errors
            ),
            "external_variance_mean": mean(external_variances),
            "absolute_tool_error_mean": mean(absolute_tool_errors),
            "selection_grid": {
                f"top_{fraction:.3f}": _top_fraction_summary(
                    labels, gains, scores, fraction=fraction
                )
                for fraction in (0.005, 0.01, 0.02, 0.05, 0.10)
            },
        }
    return {"fold": fold, "tasks": tasks}


def _top_fraction_summary(
    labels: list[int], gains: list[float], scores: list[float], *, fraction: float
) -> dict:
    count = max(1, round(fraction * len(scores)))
    selected = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:count]
    return {
        "selected_count": count,
        "precision": mean(labels[index] for index in selected),
        "selected_mean_gain": mean(gains[index] for index in selected),
        "population_gain_before_tool_cost": sum(gains[index] for index in selected)
        / len(scores),
    }


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _population_variance(values: list[float]) -> float:
    center = mean(values)
    return mean((value - center) ** 2 for value in values)


def _markdown(summary: dict) -> str:
    lines = [
        "# Real Scientific Tool Signal Diagnostic",
        "",
        "Post-hoc diagnostic on consumed folds; it is not confirmatory evidence.",
        "Tool replicas and train-only tool models are frozen before each test fold.",
        "",
        "| fold | task | worthy rate | AUROC | AP | gain corr. | ext-var/error Spearman | best positive prefix | prefix precision | prefix population gain |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        for task, values in row["tasks"].items():
            auc = values["roc_auc_worthiness"]
            average_precision = values["average_precision_worthiness"]
            positive_prefixes = [
                (name, prefix)
                for name, prefix in values["selection_grid"].items()
                if prefix["population_gain_before_tool_cost"] > 0.0
            ]
            if positive_prefixes:
                best_name, best_prefix = max(
                    positive_prefixes,
                    key=lambda item: item[1]["population_gain_before_tool_cost"],
                )
            else:
                best_name, best_prefix = "none", {
                    "precision": 0.0,
                    "population_gain_before_tool_cost": 0.0,
                }
            lines.append(
                f"| {row['fold']} | `{task}` | {values['worthy_fraction']:.4f} | "
                f"{auc if auc is not None else 'NA'} | "
                f"{average_precision if average_precision is not None else 'NA'} | "
                f"{values['gain_score_pearson']:+.4f} | "
                f"{values['external_variance_error_spearman']:+.4f} | "
                f"`{best_name}` | {best_prefix['precision']:.4f} | "
                f"{best_prefix['population_gain_before_tool_cost']:+.6f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
