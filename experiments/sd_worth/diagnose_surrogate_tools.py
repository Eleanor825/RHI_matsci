from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from harness_matsci.historical import MAIN_MATERIAL_TASKS, grouped_four_way_split, load_historical_tasks
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    gain_targets,
)
from harness_matsci.sequential_tool_benchmark import (
    TrainOnlyStructuredSurrogateToolEnvironment,
    TrainOnlySurrogateToolEnvironment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replicas", default="101,211,307")
    parser.add_argument(
        "--environment-kind",
        choices=("generic", "structured"),
        default="generic",
    )
    args = parser.parse_args()
    replicas = tuple(int(value) for value in args.replicas.split(",") if value)
    records_by_task = load_historical_tasks(args.raw_data_dir, MAIN_MATERIAL_TASKS)
    splits_by_task = {
        task: grouped_four_way_split(records, seed=args.seed)
        for task, records in records_by_task.items()
    }
    splits = {
        split: [
            record
            for task in MAIN_MATERIAL_TASKS
            for record in splits_by_task[task][split]
        ]
        for split in ("train", "feedback", "acceptance", "test")
    }
    normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
    reservations = BudgetReservationValues.fit(
        splits["train"], normalizer, GainConfig(), budget_fraction=0.10, outside_option=0.0
    )
    targets = {
        target.record_id: target
        for split in splits
        for target in gain_targets(splits[split], normalizer, GainConfig(), reservations)
    }
    environment_class = (
        TrainOnlyStructuredSurrogateToolEnvironment
        if args.environment_kind == "structured"
        else TrainOnlySurrogateToolEnvironment
    )
    environment = environment_class.fit(
        args.raw_data_dir, splits["train"], environment_seed=args.seed
    )
    result = {
        "experiment": "h9_train_only_surrogate_diagnostic",
        "seed": args.seed,
        "environment_kind": args.environment_kind,
        "replicas": replicas,
        "audit": environment.audit_nontraining_outcome_invariance(
            splits["test"], evidence_replica=replicas[0], run_seed=args.seed + 1000
        ),
        "tasks": {},
    }
    for task in MAIN_MATERIAL_TASKS:
        rows = splits_by_task[task]["test"]
        true_margin = np.asarray([environment.latent_margin(record) for record in rows])
        worthy = np.asarray([targets[record.record_id].worthy for record in rows])
        result["tasks"][task] = {}
        for stage in ("screening", "verification"):
            matrix = np.asarray(
                [
                    [
                        environment.observe(
                            record,
                            stage=stage,
                            evidence_replica=replica,
                            run_seed=args.seed + 1000,
                        ).record.features["tool_estimated_margin"]
                        for replica in replicas
                    ]
                    for record in rows
                ],
                dtype=float,
            )
            prediction = matrix.mean(axis=1)
            result["tasks"][task][stage] = {
                "n": len(rows),
                "margin_pearson": _correlation(prediction, true_margin),
                "margin_mae": float(np.mean(np.abs(prediction - true_margin))),
                "margin_sign_accuracy": float(np.mean((prediction > 0) == (true_margin > 0))),
                "worthiness_auc": _auc(worthy, prediction),
                "external_replica_variance_mean": float(np.mean(np.var(matrix, axis=1))),
                "top10_worthy_rate": _top_fraction_rate(prediction, worthy, 0.10),
                "base_worthy_rate": float(np.mean(worthy)),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _top_fraction_rate(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    selected = np.argsort(-scores)[:count]
    return float(np.mean(labels[selected]))


if __name__ == "__main__":
    main()
