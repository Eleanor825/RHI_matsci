from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from harness_matsci.conformal_gain import ConformalGainObservation, fit_split_conformal_gain
from harness_matsci.metrics import binary_metrics
from harness_matsci.risk_control import audit_risk_controlled_grid
from harness_matsci.schema import ActionRecord
from harness_matsci.safe_evidence_residual import (
    EvidenceResidualPrediction,
    EvidenceResidualState,
    TemperatureCalibratedEvidenceModel,
    fit_evidence_temperature,
    fit_safe_aggregate_evidence_residual,
)
from harness_matsci.sequential_safe_evidence import (
    build_sequential_evidence_states,
    load_worthiness_probability_caches,
)
from harness_matsci.training import train_gate_with_features


SOURCE_CONTRACTS = {
    "screening": ("screening",),
    "verification": ("verification",),
    "screening+verification": ("screening", "verification"),
}
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
REGULARIZATIONS = (0.1, 1.0, 10.0)
RISK_COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)
RISK_ALPHAS = (0.10, 0.20, 0.30, 0.40)
STATIC_RELIABILITY_FEATURES = (
    "verbal_confidence",
    "perturbation_stability",
    "evidence_support",
    "evidence_conflict",
    "source_reliability",
    "source_risk",
    "ood_score",
    "estimate_agreement",
    "extraction_confidence",
    "surrogate_uncertainty",
    "cost",
    "reversibility",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        default="gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    trajectories = _load_jsonl(Path(args.trajectories))
    models = tuple(value for value in args.models.split(",") if value)
    cache_paths = {
        model: Path(args.cache_dir) / f"{model}.json"
        for model in models
    }
    probabilities = load_worthiness_probability_caches(
        trajectories,
        model_cache_paths=cache_paths,
    )
    report = run_experiment(
        trajectories,
        probabilities,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["test_summary"], ensure_ascii=False, sort_keys=True))


def run_experiment(
    trajectories: Sequence[dict[str, object]],
    llm_probabilities: dict[str, tuple[float, ...]],
    *,
    cv_folds: int = 5,
    seed: int = 1729,
) -> dict[str, object]:
    split_rows = {
        split: [row for row in trajectories if row["split"] == split]
        for split in ("train", "feedback", "acceptance", "test")
    }
    if any(not rows for rows in split_rows.values()):
        raise ValueError("train, feedback, acceptance, and test must all be non-empty")
    state_grid = {
        contract: {
            split: build_sequential_evidence_states(
                rows,
                llm_probabilities,
                source_stages=source_stages,
            )
            for split, rows in split_rows.items()
        }
        for contract, source_stages in SOURCE_CONTRACTS.items()
    }
    selection = _select_configuration(
        state_grid,
        cv_folds=cv_folds,
        seed=seed,
    )
    contract = str(selection["selected"]["source_contract"])
    blend_weight = float(selection["selected"]["blend_weight"])
    regularization = float(selection["selected"]["regularization"])
    states = state_grid[contract]
    base_model = fit_safe_aggregate_evidence_residual(
        states["train"],
        blend_weight=blend_weight,
        regularization=regularization,
    )
    probability_feedback, conformal_feedback = _split_feedback_states(
        states["feedback"], seed=seed
    )
    model = fit_evidence_temperature(base_model, probability_feedback)
    predictions = {
        split: model.predict(split_states)
        for split, split_states in states.items()
    }

    direct_temperature = _fit_scalar_temperature(
        [_mean(state.model_success_probabilities) for state in probability_feedback],
        [_label(state) for state in probability_feedback],
    )
    direct_probabilities = {
        split: [
            _temperature_scale(_mean(state.model_success_probabilities), direct_temperature)
            for state in split_states
        ]
        for split, split_states in states.items()
    }
    reliability_records = {
        split: _static_reliability_records(rows)
        for split, rows in split_rows.items()
    }
    feedback_calibration_ids = {state.record_id for state in probability_feedback}
    reliability_feedback_fit = [
        record
        for record in reliability_records["feedback"]
        if record.record_id in feedback_calibration_ids
    ]
    observed_reliability_features = tuple(
        feature
        for feature in STATIC_RELIABILITY_FEATURES
        if any(feature in record.features for record in reliability_records["train"])
    )
    reliability_gate = train_gate_with_features(
        reliability_records["train"],
        reliability_feedback_fit,
        feature_names=list(observed_reliability_features),
        epochs=500,
        learning_rate=0.08,
        l2=0.01,
        balance_benchmarks=True,
    )
    raw_reliability_probabilities = {
        split: reliability_gate.predict_proba(records)
        for split, records in reliability_records.items()
    }
    reliability_temperature = _fit_scalar_temperature(
        [
            value
            for record, value in zip(
                reliability_records["feedback"],
                raw_reliability_probabilities["feedback"],
            )
            if record.record_id in feedback_calibration_ids
        ],
        [_label(state) for state in probability_feedback],
    )
    reliability_probabilities = {
        split: [
            _temperature_scale(value, reliability_temperature)
            for value in values
        ]
        for split, values in raw_reliability_probabilities.items()
    }

    conformal = fit_split_conformal_gain(
        [
            ConformalGainObservation(
                record_id=state.record_id,
                predicted_mean_gain=prediction.expected_net_gain,
                predicted_scale=math.sqrt(max(1e-12, prediction.total_variance)),
                observed_gain=state.actual_net_gain,
            )
            for state, prediction in zip(
                conformal_feedback,
                model.predict(conformal_feedback),
            )
        ],
        alpha=0.10,
    )
    lower_gain_scores = {
        split: [
            conformal.predict_one(
                record_id=state.record_id,
                predicted_mean_gain=prediction.expected_net_gain,
                predicted_scale=math.sqrt(max(1e-12, prediction.total_variance)),
            ).lower_gain_bound
            for state, prediction in zip(states[split], predictions[split])
        ]
        for split in states
    }

    tool_probabilities = {
        stage: {
            split: _tool_probabilities(split_rows[split], stage)
            for split in split_rows
        }
        for stage in ("screening", "verification")
    }
    tool_temperatures = {
        stage: _fit_scalar_temperature(
            [
                value
                for state, value in zip(
                    states["feedback"], tool_probabilities[stage]["feedback"]
                )
                if state.record_id in {row.record_id for row in probability_feedback}
            ],
            [_label(state) for state in probability_feedback],
        )
        for stage in tool_probabilities
    }
    calibrated_tool_probabilities = {
        stage: {
            split: [
                _temperature_scale(value, tool_temperatures[stage])
                for value in values
            ]
            for split, values in by_split.items()
        }
        for stage, by_split in tool_probabilities.items()
    }

    method_scores = {
        "direct_llm": direct_probabilities,
        "static_reliability": reliability_probabilities,
        "static_screening_tool": calibrated_tool_probabilities["screening"],
        "static_verification_tool": calibrated_tool_probabilities["verification"],
        "ours_action_worthiness": {
            split: [prediction.action_worthiness for prediction in split_predictions]
            for split, split_predictions in predictions.items()
        },
        "ours_conformal_lower_gain": lower_gain_scores,
    }
    certificates = {
        method: {
            f"alpha_{alpha:.2f}": _risk_certificate(
                scores["acceptance"],
                [_label(state) for state in states["acceptance"]],
                alpha=alpha,
            )
            for alpha in RISK_ALPHAS
        }
        for method, scores in method_scores.items()
    }
    deployments = {
        method: {
            key: _deploy_certificate(
                certificate,
                method_scores[method]["test"],
                states["test"],
            )
            for key, certificate in by_alpha.items()
        }
        for method, by_alpha in certificates.items()
    }
    test_metrics = {
        method: _probability_metrics(states["test"], scores["test"])
        for method, scores in method_scores.items()
        if method != "ours_conformal_lower_gain"
    }
    gain_metrics = _gain_metrics(states["test"], predictions["test"])
    conformal_coverage = _conformal_coverage(
        states["test"], predictions["test"], conformal
    )
    by_task = {
        task: {
            method: _probability_metrics(
                [state for state in states["test"] if state.benchmark == task],
                [
                    score
                    for state, score in zip(states["test"], scores["test"])
                    if state.benchmark == task
                ],
            )
            for method, scores in method_scores.items()
            if method != "ours_conformal_lower_gain"
        }
        for task in sorted({state.benchmark for state in states["test"]})
    }
    return {
        "schema_version": "safe-crossed-evidence-residual-sequential-v1",
        "protocol": {
            "data_roles": {
                "train": "group-cross-validated method selection and final fitting",
                "feedback_probability": "temperature calibration only",
                "feedback_conformal": "one-sided normalized split-conformal gain only",
                "acceptance": "risk certificate and harness acceptance only",
                "test": "single held-out evaluation after all choices are frozen",
            },
            "feedback_record_ids": {
                "probability": [state.record_id for state in probability_feedback],
                "conformal": [state.record_id for state in conformal_feedback],
            },
            "risk_coverages": list(RISK_COVERAGES),
            "risk_alphas": list(RISK_ALPHAS),
            "cv_folds": cv_folds,
            "seed": seed,
        },
        "selection": selection,
        "model": model.to_json(),
        "direct_temperature": direct_temperature,
        "static_reliability": {
            "features": list(observed_reliability_features),
            "temperature": reliability_temperature,
            "gate": reliability_gate.to_json(),
        },
        "tool_temperatures": tool_temperatures,
        "conformal": conformal.to_json(),
        "test_summary": {
            "probability_metrics": test_metrics,
            "gain_metrics": gain_metrics,
            "conformal_coverage": conformal_coverage,
            "by_task": by_task,
            "certificates": certificates,
            "deployments": deployments,
        },
        "test_predictions": [
            {
                "record_id": state.record_id,
                "benchmark": state.benchmark,
                "worthy": _label(state),
                "actual_net_gain": state.actual_net_gain,
                "direct_probability": direct_probabilities["test"][index],
                "ours": prediction.to_json(),
                "conformal_lower_gain": lower_gain_scores["test"][index],
            }
            for index, (state, prediction) in enumerate(
                zip(states["test"], predictions["test"])
            )
        ],
    }


def _select_configuration(state_grid, *, cv_folds: int, seed: int):
    reports = []
    for contract, by_split in state_grid.items():
        states = by_split["train"]
        folds = _group_folds(states, cv_folds=cv_folds, seed=seed)
        for blend_weight in BLEND_WEIGHTS:
            for regularization in REGULARIZATIONS:
                labels = []
                probabilities = []
                gains = []
                predicted_gains = []
                fold_reports = []
                for held_groups in folds:
                    fit_states = [
                        state
                        for state in states
                        if _group_key(state) not in held_groups
                    ]
                    held_states = [
                        state
                        for state in states
                        if _group_key(state) in held_groups
                    ]
                    model = fit_safe_aggregate_evidence_residual(
                        fit_states,
                        blend_weight=blend_weight,
                        regularization=regularization,
                    )
                    predictions = model.predict(held_states)
                    fold_labels = [_label(state) for state in held_states]
                    fold_probabilities = [row.action_worthiness for row in predictions]
                    labels.extend(fold_labels)
                    probabilities.extend(fold_probabilities)
                    gains.extend(state.actual_net_gain for state in held_states)
                    predicted_gains.extend(row.expected_net_gain for row in predictions)
                    fold_reports.append(
                        {
                            "held_group_count": len(held_groups),
                            "held_record_count": len(held_states),
                            "metrics": binary_metrics(
                                fold_labels, fold_probabilities, threshold=0.5
                            ),
                        }
                    )
                metrics = binary_metrics(labels, probabilities, threshold=0.5)
                gain_mae = _mean(
                    abs(actual - predicted)
                    for actual, predicted in zip(gains, predicted_gains)
                )
                reports.append(
                    {
                        "source_contract": contract,
                        "blend_weight": blend_weight,
                        "regularization": regularization,
                        "selection_score": metrics["log_loss"],
                        "metrics": metrics,
                        "gain_mae": gain_mae,
                        "folds": fold_reports,
                    }
                )
    selected = min(
        reports,
        key=lambda row: (
            row["selection_score"],
            row["metrics"]["brier"],
            row["gain_mae"],
            row["source_contract"],
            row["blend_weight"],
            row["regularization"],
        ),
    )
    return {
        "criterion": "minimum group-held-out binary log loss; Brier and gain MAE are deterministic tie-breakers",
        "selected": selected,
        "candidates": reports,
    }


def _group_folds(states, *, cv_folds: int, seed: int):
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two")
    groups = sorted({_group_key(state) for state in states})
    folds = [set() for _ in range(cv_folds)]
    for group in groups:
        digest = hashlib.sha256(f"{seed}::{group}".encode()).hexdigest()
        folds[int(digest[:16], 16) % cv_folds].add(group)
    if any(not fold for fold in folds):
        raise ValueError("group hash produced an empty cross-validation fold")
    return folds


def _split_feedback_states(states, *, seed: int):
    groups = sorted({_group_key(state) for state in states})
    calibration_groups = {
        group
        for group in groups
        if int(hashlib.sha256(f"feedback::{seed}::{group}".encode()).hexdigest()[:16], 16) % 2 == 0
    }
    if not calibration_groups or len(calibration_groups) == len(groups):
        calibration_groups = set(groups[::2])
    calibration = [state for state in states if _group_key(state) in calibration_groups]
    conformal = [state for state in states if _group_key(state) not in calibration_groups]
    if not calibration or not conformal:
        raise ValueError("feedback split must produce two non-empty group-disjoint subsets")
    return calibration, conformal


def _risk_certificate(scores, labels, *, alpha: float, delta: float = 0.05):
    audit = audit_risk_controlled_grid(
        list(scores),
        list(labels),
        coverages=RISK_COVERAGES,
        alpha=alpha,
        delta=delta,
    )
    certified = [row for row in audit if row["certified"]]
    policy = max(certified, key=lambda row: row["selected_count"]) if certified else None
    return {
        "alpha": alpha,
        "delta": delta,
        "familywise_method": "Bonferroni exact Clopper-Pearson over frozen coverage grid",
        "certified": policy is not None,
        "selected_policy": policy,
        "audit": audit,
    }


def _deploy_certificate(certificate, scores, states):
    policy = certificate["selected_policy"]
    if policy is None:
        selected = []
    else:
        selected = [
            index for index, score in enumerate(scores)
            if score >= float(policy["threshold"])
        ]
    gains = [states[index].actual_net_gain for index in selected]
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(states),
        "selective_risk": _mean(float(value <= 0.0) for value in gains),
        "selected_mean_net_gain": _mean(gains),
        "population_net_gain": sum(gains) / len(states),
    }


def _probability_metrics(states, probabilities):
    return binary_metrics(
        [_label(state) for state in states],
        list(probabilities),
        threshold=0.5,
    )


def _gain_metrics(states, predictions):
    actual = [state.actual_net_gain for state in states]
    predicted = [row.expected_net_gain for row in predictions]
    squared_errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predicted, actual)
    ]
    return {
        "mae": _mean(abs(prediction - target) for prediction, target in zip(predicted, actual)),
        "rmse": math.sqrt(_mean(squared_errors)),
        "correlation": _correlation(predicted, actual),
        "internal_variance_error_correlation": _correlation(
            [row.internal_variance for row in predictions], squared_errors
        ),
        "external_variance_error_correlation": _correlation(
            [row.external_variance for row in predictions], squared_errors
        ),
        "interaction_variance_error_correlation": _correlation(
            [row.interaction_variance for row in predictions], squared_errors
        ),
        "mean_internal_variance": _mean(row.internal_variance for row in predictions),
        "mean_external_variance": _mean(row.external_variance for row in predictions),
        "mean_interaction_variance": _mean(row.interaction_variance for row in predictions),
        "mean_irreducible_variance": _mean(row.irreducible_variance for row in predictions),
    }


def _conformal_coverage(states, predictions, calibrator):
    covered = []
    for state, prediction in zip(states, predictions):
        bound = calibrator.predict_one(
            record_id=state.record_id,
            predicted_mean_gain=prediction.expected_net_gain,
            predicted_scale=math.sqrt(max(1e-12, prediction.total_variance)),
        )
        covered.append(state.actual_net_gain >= bound.lower_gain_bound)
    return {
        "target": 1.0 - calibrator.alpha,
        "empirical": _mean(float(value) for value in covered),
        "count": len(covered),
    }


def _tool_probabilities(trajectories, stage):
    output = []
    for trajectory in trajectories:
        matches = [row for row in trajectory["stages"] if row["stage"] == stage]
        if len(matches) != 1:
            raise ValueError(f"missing unique {stage} stage")
        output.append(
            float(
                matches[0]["visible_observation"]["record"]["features"][
                    "tool_positive_probability"
                ]
            )
        )
    return output


def _static_reliability_records(trajectories):
    output = []
    for trajectory in trajectories:
        base = next(row for row in trajectory["stages"] if row["stage"] == "base")
        payload = dict(base["visible_observation"]["record"])
        hidden = trajectory["hidden_outcome_for_evaluation_only"]
        payload["label"] = int(hidden["success_label"])
        payload["utility"] = float(hidden["normalized_utility"])
        output.append(ActionRecord.from_json(payload))
    return output


def _fit_scalar_temperature(probabilities, labels):
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("temperature calibration inputs must be non-empty and aligned")
    best_temperature = 1.0
    best_loss = float("inf")
    for step in range(2, 201):
        temperature = step / 20.0
        calibrated = [_temperature_scale(value, temperature) for value in probabilities]
        loss = -_mean(
            label * math.log(_clip(probability))
            + (1 - label) * math.log(_clip(1.0 - probability))
            for label, probability in zip(labels, calibrated)
        )
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _temperature_scale(probability, temperature):
    return _sigmoid(_logit(probability) / temperature)


def _group_key(state: EvidenceResidualState):
    return f"{state.benchmark}::{state.candidate_group_ids[0]}"


def _label(state: EvidenceResidualState):
    return int(state.actual_net_gain > 0.0)


def _correlation(left, right):
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _clip(value):
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _logit(value):
    probability = _clip(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
