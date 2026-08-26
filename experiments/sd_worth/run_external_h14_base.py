from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from harness_matsci.continuous_gain_calibration import (
    ContinuousGainObservation,
    fit_continuous_gain_calibrator,
)
from harness_matsci.hurdle_gain_calibration import (
    HurdleGainObservation,
    fit_hurdle_gain_calibrator,
)
from harness_matsci.risk_control import (
    empirical_bernstein_lower_confidence,
    select_risk_controlled_grid,
)
from harness_matsci.schema import ActionRecord
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    gain_targets,
)
from harness_matsci.sequential_tool_experiment import GainReplicaEnsemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1401)
    args = parser.parse_args()
    result = run_external_base(args.fold_dir, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


def run_external_base(fold_dir: str | Path, *, seed: int) -> dict:
    fold_root = Path(fold_dir)
    tasks = (
        "external_pairwise_dielectric",
        "external_unique_jdft2d",
        "external_extreme_phonons",
    )
    splits = {
        split: [
            record
            for task in tasks
            for record in _load_records(fold_root / f"{task}.{split}.jsonl")
        ]
        for split in ("train", "feedback", "acceptance", "test")
    }
    config = GainConfig(utility_mode="continuous_realized")
    normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
    reservations = BudgetReservationValues.fit(
        splits["train"],
        normalizer,
        config,
        budget_fraction=0.10,
        outside_option=0.0,
        strategy="potential_success_slot_value",
    )
    targets = {
        split: gain_targets(records, normalizer, config, reservations)
        for split, records in splits.items()
    }
    target_by_id = {
        target.record_id: target
        for split_targets in targets.values()
        for target in split_targets
    }
    model = GainReplicaEnsemble.fit(
        splits["train"],
        [target.gain for target in targets["train"]],
        seed=seed,
        replicas=5,
        trees=200,
    )
    replica_summaries = {
        split: _replica_summaries(model.predict_matrix(records), records)
        for split, records in splits.items()
        if split != "train"
    }
    continuous_feedback = _continuous_observations(
        replica_summaries["feedback"], target_by_id
    )
    gaussian = fit_continuous_gain_calibrator(
        continuous_feedback, use_source_variance=False
    )
    hurdle_feedback = _hurdle_observations(
        replica_summaries["feedback"], target_by_id
    )
    hurdle = fit_hurdle_gain_calibrator(
        hurdle_feedback,
        use_source_variance=False,
        beta=0.10,
        tail_weight=0.50,
    )
    methods = {
        "verbal_confidence": {
            split: [record.features["verbal_confidence"] for record in splits[split]]
            for split in ("acceptance", "test")
        },
        "evidence_heuristic": {
            split: [
                record.features["evidence_support"]
                - record.features["evidence_conflict"]
                for record in splits[split]
            ]
            for split in ("acceptance", "test")
        },
        "ensemble_mean": {
            split: [row["mean_gain"] for row in replica_summaries[split]]
            for split in ("acceptance", "test")
        },
        "ensemble_lcb": {
            split: [
                row["mean_gain"] - math.sqrt(max(0.0, row["internal_variance"]))
                for row in replica_summaries[split]
            ]
            for split in ("acceptance", "test")
        },
        "gaussian_p_positive": {
            split: [
                prediction.action_worthiness
                for prediction in gaussian.predict(
                    _continuous_observations(replica_summaries[split], target_by_id)
                )
            ]
            for split in ("acceptance", "test")
        },
    }
    for tail_weight in (0.0, 0.25, 0.50, 1.0):
        fitted = replace(hurdle, tail_weight=tail_weight)
        predictions_by_split = {
            split: fitted.predict(
                _hurdle_observations(replica_summaries[split], target_by_id)
            )
            for split in ("acceptance", "test")
        }
        methods[f"hurdle_p_tail_{tail_weight:.2f}"] = {
            split: [prediction.p_positive_gain for prediction in predictions]
            for split, predictions in predictions_by_split.items()
        }
        methods[f"hurdle_mean_tail_{tail_weight:.2f}"] = {
            split: [prediction.mean_gain for prediction in predictions]
            for split, predictions in predictions_by_split.items()
        }
        methods[f"hurdle_score_tail_{tail_weight:.2f}"] = {
            split: [prediction.tail_risk_score for prediction in predictions]
            for split, predictions in predictions_by_split.items()
        }
    labels = {
        split: [target.worthy for target in targets[split]]
        for split in ("acceptance", "test")
    }
    gains = {
        split: [target.gain for target in targets[split]]
        for split in ("acceptance", "test")
    }
    results = {
        method: _evaluate_method(
            scores,
            labels,
            gains,
            splits,
        )
        for method, scores in methods.items()
    }
    return {
        "experiment": "h14_external_base_development",
        "development_only": True,
        "fold_dir": str(fold_root),
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "reservation_values": reservations.to_json(),
        "gain_config": {
            "utility_mode": config.utility_mode,
            "cost_weight": config.cost_weight,
            "failure_harm_weight": config.failure_harm_weight,
        },
        "methods": results,
    }


def _evaluate_method(scores, labels, gains, splits) -> dict:
    selection = select_risk_controlled_grid(
        scores["acceptance"],
        labels["acceptance"],
        coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
        alpha=0.20,
        delta=0.05,
    )
    acceptance = _evaluate_threshold(
        scores["acceptance"], gains["acceptance"], labels["acceptance"], selection.threshold
    )
    contributions = [
        gain if score >= selection.threshold else 0.0
        for score, gain in zip(scores["acceptance"], gains["acceptance"])
    ]
    gain_lcb = empirical_bernstein_lower_confidence(
        contributions,
        0.05 / 8.0,
        lower_bound=-1.40,
        upper_bound=1.00,
    )
    risk_only_active = selection.selected_count > 0 and acceptance["population_gain"] > 0.0
    joint_active = selection.selected_count > 0 and gain_lcb > 0.0
    test = (
        _evaluate_threshold(scores["test"], gains["test"], labels["test"], selection.threshold)
        if risk_only_active
        else _empty_metrics(len(scores["test"]))
    )
    joint_test = (
        _evaluate_threshold(scores["test"], gains["test"], labels["test"], selection.threshold)
        if joint_active
        else _empty_metrics(len(scores["test"]))
    )
    return {
        "risk_selection": selection.to_json(),
        "acceptance": acceptance,
        "acceptance_gain_lcb": gain_lcb,
        "risk_only_active": risk_only_active,
        "joint_certificate_active": joint_active,
        "test": test,
        "joint_certificate_test": joint_test,
    }


def _evaluate_threshold(scores, gains, labels, threshold) -> dict:
    selected = [index for index, score in enumerate(scores) if score >= threshold]
    if not selected:
        return _empty_metrics(len(scores))
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(scores),
        "selective_risk": sum(1 - labels[index] for index in selected) / len(selected),
        "selected_mean_gain": sum(gains[index] for index in selected) / len(selected),
        "population_gain": sum(gains[index] for index in selected) / len(scores),
    }


def _empty_metrics(population: int) -> dict:
    return {
        "selected_count": 0,
        "coverage": 0.0,
        "selective_risk": 0.0,
        "selected_mean_gain": 0.0,
        "population_gain": 0.0,
    }


def _replica_summaries(matrix: list[list[float]], records: list[ActionRecord]) -> list[dict]:
    output = []
    for index, record in enumerate(records):
        values = [replica[index] for replica in matrix]
        center = sum(values) / len(values)
        output.append(
            {
                "record_id": record.record_id,
                "benchmark": record.benchmark,
                "mean_gain": center,
                "internal_variance": sum((value - center) ** 2 for value in values)
                / len(values),
            }
        )
    return output


def _continuous_observations(rows, target_by_id):
    return [
        ContinuousGainObservation(
            record_id=row["record_id"],
            benchmark=row["benchmark"],
            mean_gain=row["mean_gain"],
            internal_variance=row["internal_variance"],
            external_variance=0.0,
            gain=target_by_id[row["record_id"]].gain,
        )
        for row in rows
    ]


def _hurdle_observations(rows, target_by_id):
    return [
        HurdleGainObservation(
            record_id=row["record_id"],
            benchmark=row["benchmark"],
            mean_gain=row["mean_gain"],
            internal_variance=row["internal_variance"],
            external_variance=0.0,
            gain=target_by_id[row["record_id"]].gain,
        )
        for row in rows
    ]


def _load_records(path: Path) -> list[ActionRecord]:
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _markdown(result: dict) -> str:
    lines = [
        "# H14 External Base Development Result",
        "",
        "Fold 0 only. No external tool acquisition or source decomposition is used.",
        "",
        "| method | policy | coverage | risk | selected gain | population gain | acceptance gain LCB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, values in result["methods"].items():
        test = values["test"]
        policy = "active" if values["risk_only_active"] else "no_op"
        lines.append(
            f"| `{method}` | `{policy}` | {test['coverage']:.4f} | "
            f"{test['selective_risk']:.4f} | {test['selected_mean_gain']:+.4f} | "
            f"{test['population_gain']:+.6f} | {values['acceptance_gain_lcb']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
