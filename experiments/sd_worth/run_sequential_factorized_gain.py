from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from harness_matsci.conformal_gain import ConformalGainObservation, fit_split_conformal_gain
from harness_matsci.factorized_gain_harness import (
    FactorizedGainState,
    fit_factorized_gain_harness,
    fit_factorized_temperature,
)
from harness_matsci.metrics import binary_metrics
from harness_matsci.risk_control import audit_risk_controlled_grid, binomial_upper_confidence
from harness_matsci.schema import ActionRecord
from harness_matsci.sequential_safe_evidence import (
    build_factorized_gain_states,
    load_worthiness_prediction_caches,
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
RISK_ALPHAS = (0.10, 0.20, 0.30, 0.40, 0.50)
FIXED_TASK_BUDGET = 0.10
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
    parser.add_argument("--fresh-relative-to")
    parser.add_argument(
        "--models",
        default="gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2729)
    args = parser.parse_args()
    trajectories = _load_jsonl(Path(args.trajectories))
    prior_ids = set()
    if args.fresh_relative_to:
        prior_ids = {
            row["record_id"] for row in _load_jsonl(Path(args.fresh_relative_to))
        }
    models = tuple(value for value in args.models.split(",") if value)
    predictions = load_worthiness_prediction_caches(
        trajectories,
        model_cache_paths={
            model: Path(args.cache_dir) / f"{model}.json"
            for model in models
        },
    )
    report = run_experiment(
        trajectories,
        predictions,
        prior_ids=prior_ids,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))


def run_experiment(trajectories, llm_predictions, *, prior_ids, cv_folds, seed):
    split_rows = {
        split: [row for row in trajectories if row["split"] == split]
        for split in ("train", "feedback", "acceptance", "test")
    }
    evaluation_rows = {
        split: [
            row for row in split_rows[split]
            if not prior_ids or row["record_id"] not in prior_ids
        ]
        for split in ("acceptance", "test")
    }
    if any(not rows for rows in split_rows.values()) or any(
        not rows for rows in evaluation_rows.values()
    ):
        raise ValueError("all data roles must be non-empty")
    states_by_contract = {
        contract: {
            "train": build_factorized_gain_states(
                split_rows["train"], llm_predictions, source_stages=sources
            ),
            "feedback": build_factorized_gain_states(
                split_rows["feedback"], llm_predictions, source_stages=sources
            ),
            "acceptance": build_factorized_gain_states(
                evaluation_rows["acceptance"], llm_predictions, source_stages=sources
            ),
            "test": build_factorized_gain_states(
                evaluation_rows["test"], llm_predictions, source_stages=sources
            ),
        }
        for contract, sources in SOURCE_CONTRACTS.items()
    }
    selection = _select_configuration(states_by_contract, cv_folds=cv_folds, seed=seed)
    chosen = selection["selected"]
    states = states_by_contract[chosen["source_contract"]]
    base_model = fit_factorized_gain_harness(
        states["train"],
        blend_weight=chosen["blend_weight"],
        regularization=chosen["regularization"],
    )
    probability_feedback, conformal_feedback = _split_feedback(states["feedback"], seed)
    model = fit_factorized_temperature(base_model, probability_feedback)
    predictions = {split: model.predict(rows) for split, rows in states.items()}
    conformal = fit_split_conformal_gain(
        [
            ConformalGainObservation(
                state.record_id,
                prediction.expected_net_gain,
                math.sqrt(max(1e-12, prediction.total_gain_variance)),
                state.actual_net_gain,
            )
            for state, prediction in zip(
                conformal_feedback,
                model.predict(conformal_feedback),
            )
        ],
        alpha=0.10,
    )
    lower_gain = {
        split: [
            conformal.predict_one(
                record_id=state.record_id,
                predicted_mean_gain=prediction.expected_net_gain,
                predicted_scale=math.sqrt(max(1e-12, prediction.total_gain_variance)),
            ).lower_gain_bound
            for state, prediction in zip(states[split], predictions[split])
        ]
        for split in states
    }
    direct_temperature = _fit_temperature(
        [_mean(state.model_worthiness_probabilities) for state in probability_feedback],
        [_worthy(state) for state in probability_feedback],
    )
    direct = {
        split: [
            _temperature_scale(_mean(state.model_worthiness_probabilities), direct_temperature)
            for state in rows
        ]
        for split, rows in states.items()
    }
    reliability_rows = {
        "train": _static_records(split_rows["train"]),
        "feedback": _static_records(split_rows["feedback"]),
        "acceptance": _static_records(evaluation_rows["acceptance"]),
        "test": _static_records(evaluation_rows["test"]),
    }
    calibration_ids = {state.record_id for state in probability_feedback}
    reliability_feedback = [
        row for row in reliability_rows["feedback"] if row.record_id in calibration_ids
    ]
    feature_names = [
        name
        for name in STATIC_RELIABILITY_FEATURES
        if any(name in row.features for row in reliability_rows["train"])
    ]
    gate = train_gate_with_features(
        reliability_rows["train"],
        reliability_feedback,
        feature_names=feature_names,
        epochs=500,
        learning_rate=0.08,
        l2=0.01,
        balance_benchmarks=True,
    )
    raw_reliability = {
        split: gate.predict_proba(rows) for split, rows in reliability_rows.items()
    }
    reliability_temperature = _fit_temperature(
        [
            score
            for row, score in zip(reliability_rows["feedback"], raw_reliability["feedback"])
            if row.record_id in calibration_ids
        ],
        [_worthy(state) for state in probability_feedback],
    )
    reliability = {
        split: [_temperature_scale(score, reliability_temperature) for score in scores]
        for split, scores in raw_reliability.items()
    }
    scores = {
        "direct_llm": direct,
        "static_reliability": reliability,
        "factorized_worthiness": {
            split: [prediction.action_worthiness for prediction in rows]
            for split, rows in predictions.items()
        },
        "factorized_conformal_lower_gain": lower_gain,
    }
    certificates = {
        method: {
            f"alpha_{alpha:.2f}": _certificate(
                values["acceptance"],
                [_worthy(state) for state in states["acceptance"]],
                alpha,
            )
            for alpha in RISK_ALPHAS
        }
        for method, values in scores.items()
    }
    fixed_budget_certificates = {
        method: {
            f"alpha_{alpha:.2f}": _fixed_task_budget_certificate(
                values["acceptance"],
                states["acceptance"],
                alpha=alpha,
            )
            for alpha in RISK_ALPHAS
        }
        for method, values in scores.items()
    }
    deployments = {
        method: {
            alpha: _deploy(certificate, scores[method]["test"], states["test"])
            for alpha, certificate in rows.items()
        }
        for method, rows in certificates.items()
    }
    fixed_budget_deployments = {
        method: {
            alpha: _deploy_fixed_task_budget(
                certificate,
                scores[method]["test"],
                states["test"],
            )
            for alpha, certificate in rows.items()
        }
        for method, rows in fixed_budget_certificates.items()
    }
    probability_metrics = {
        method: binary_metrics(
            [_worthy(state) for state in states["test"]],
            values["test"],
            threshold=0.5,
        )
        for method, values in scores.items()
        if method != "factorized_conformal_lower_gain"
    }
    gain_metrics = _gain_metrics(states["test"], predictions["test"])
    conformal_coverage = _conformal_coverage(states["test"], predictions["test"], conformal)
    return {
        "schema_version": "factorized-success-utility-gain-harness-v1",
        "prior_record_count_excluded_from_acceptance_test": len(prior_ids),
        "sizes": {
            "train": len(states["train"]),
            "feedback": len(states["feedback"]),
            "acceptance": len(states["acceptance"]),
            "test": len(states["test"]),
        },
        "selection": selection,
        "model": model.to_json(),
        "conformal": conformal.to_json(),
        "direct_temperature": direct_temperature,
        "static_reliability": {
            "temperature": reliability_temperature,
            "features": feature_names,
            "gate": gate.to_json(),
        },
        "summary": {
            "probability_metrics": probability_metrics,
            "gain_metrics": gain_metrics,
            "conformal_coverage": conformal_coverage,
            "certificates": certificates,
            "deployments": deployments,
            "fixed_task_budget": FIXED_TASK_BUDGET,
            "fixed_budget_certificates": fixed_budget_certificates,
            "fixed_budget_deployments": fixed_budget_deployments,
        },
        "acceptance_predictions": _prediction_rows(
            states["acceptance"], predictions["acceptance"], scores, "acceptance"
        ),
        "test_predictions": _prediction_rows(
            states["test"], predictions["test"], scores, "test"
        ),
    }


def _select_configuration(states_by_contract, *, cv_folds, seed):
    candidates = []
    for contract, splits in states_by_contract.items():
        states = splits["train"]
        folds = _group_folds(states, cv_folds, seed)
        for blend in BLEND_WEIGHTS:
            for regularization in REGULARIZATIONS:
                labels = []
                probabilities = []
                gain_errors = []
                for held_groups in folds:
                    fit_rows = [state for state in states if _group_key(state) not in held_groups]
                    held_rows = [state for state in states if _group_key(state) in held_groups]
                    model = fit_factorized_gain_harness(
                        fit_rows,
                        blend_weight=blend,
                        regularization=regularization,
                    )
                    predictions = model.predict(held_rows)
                    labels.extend(_worthy(state) for state in held_rows)
                    probabilities.extend(row.action_worthiness for row in predictions)
                    gain_errors.extend(
                        abs(row.expected_net_gain - state.actual_net_gain)
                        for state, row in zip(held_rows, predictions)
                    )
                metrics = binary_metrics(labels, probabilities, threshold=0.5)
                candidates.append(
                    {
                        "source_contract": contract,
                        "blend_weight": blend,
                        "regularization": regularization,
                        "selection_score": metrics["log_loss"],
                        "gain_mae": _mean(gain_errors),
                        "metrics": metrics,
                    }
                )
    selected = min(
        candidates,
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
        "criterion": "minimum train-group-held-out worthiness log loss",
        "selected": selected,
        "candidates": candidates,
    }


def _group_folds(states, count, seed):
    groups = sorted({_group_key(state) for state in states})
    folds = [set() for _ in range(count)]
    for group in groups:
        digest = hashlib.sha256(f"{seed}::{group}".encode()).hexdigest()
        folds[int(digest[:16], 16) % count].add(group)
    if any(not fold for fold in folds):
        raise ValueError("empty group fold")
    return folds


def _split_feedback(states, seed):
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
    return calibration, conformal


def _certificate(scores, labels, alpha, delta=0.05):
    audit = audit_risk_controlled_grid(
        list(scores), list(labels), coverages=RISK_COVERAGES, alpha=alpha, delta=delta
    )
    valid = [row for row in audit if row["certified"]]
    selected = max(valid, key=lambda row: row["selected_count"]) if valid else None
    return {
        "alpha": alpha,
        "delta": delta,
        "certified": selected is not None,
        "selected_policy": selected,
        "audit": audit,
    }


def _deploy(certificate, scores, states):
    policy = certificate["selected_policy"]
    selected = [] if policy is None else [
        index for index, score in enumerate(scores) if score >= policy["threshold"]
    ]
    gains = [states[index].actual_net_gain for index in selected]
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(states),
        "selective_risk": _mean(float(gain <= 0.0) for gain in gains),
        "selected_mean_net_gain": _mean(gains),
        "population_net_gain": sum(gains) / len(states),
    }


def _fixed_task_budget_certificate(scores, states, *, alpha, delta=0.05):
    selected = _fixed_task_budget_indices(scores, states)
    errors = sum(states[index].actual_net_gain <= 0.0 for index in selected)
    upper = binomial_upper_confidence(errors, len(selected), delta)
    return {
        "policy": "top 10% independently within each scientific task",
        "budget_fraction": FIXED_TASK_BUDGET,
        "selected_count": len(selected),
        "errors": errors,
        "empirical_risk": errors / len(selected),
        "risk_upper_bound": upper,
        "alpha": alpha,
        "delta": delta,
        "certified": upper <= alpha,
    }


def _deploy_fixed_task_budget(certificate, scores, states):
    selected = (
        _fixed_task_budget_indices(scores, states)
        if certificate["certified"]
        else []
    )
    shadow_selected = _fixed_task_budget_indices(scores, states)
    gains = [states[index].actual_net_gain for index in selected]
    shadow_gains = [states[index].actual_net_gain for index in shadow_selected]
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(states),
        "selective_risk": _mean(float(gain <= 0.0) for gain in gains),
        "selected_mean_net_gain": _mean(gains),
        "population_net_gain": sum(gains) / len(states),
        "shadow_selected_count": len(shadow_selected),
        "shadow_selective_risk": _mean(
            float(gain <= 0.0) for gain in shadow_gains
        ),
        "shadow_selected_mean_net_gain": _mean(shadow_gains),
        "shadow_population_net_gain": sum(shadow_gains) / len(states),
    }


def _fixed_task_budget_indices(scores, states):
    selected = []
    for task in sorted({state.benchmark for state in states}):
        indices = [
            index for index, state in enumerate(states) if state.benchmark == task
        ]
        count = max(1, math.ceil(FIXED_TASK_BUDGET * len(indices)))
        selected.extend(
            sorted(indices, key=lambda index: (-scores[index], index))[:count]
        )
    return selected


def _gain_metrics(states, predictions):
    actual = [state.actual_net_gain for state in states]
    predicted = [row.expected_net_gain for row in predictions]
    squared = [(left - right) ** 2 for left, right in zip(predicted, actual)]
    return {
        "mae": _mean(abs(left - right) for left, right in zip(predicted, actual)),
        "rmse": math.sqrt(_mean(squared)),
        "correlation": _correlation(predicted, actual),
        "internal_variance_error_correlation": _correlation(
            [row.gain_internal_variance for row in predictions], squared
        ),
        "external_variance_error_correlation": _correlation(
            [row.gain_external_variance for row in predictions], squared
        ),
        "interaction_variance_error_correlation": _correlation(
            [row.gain_interaction_variance for row in predictions], squared
        ),
    }


def _conformal_coverage(states, predictions, calibrator):
    covered = []
    for state, prediction in zip(states, predictions):
        bound = calibrator.predict_one(
            record_id=state.record_id,
            predicted_mean_gain=prediction.expected_net_gain,
            predicted_scale=math.sqrt(max(1e-12, prediction.total_gain_variance)),
        )
        covered.append(state.actual_net_gain >= bound.lower_gain_bound)
    return {"target": 0.9, "empirical": _mean(float(value) for value in covered), "count": len(covered)}


def _prediction_rows(states, predictions, scores, split):
    return [
        {
            "record_id": state.record_id,
            "benchmark": state.benchmark,
            "worthy": _worthy(state),
            "actual_success": state.actual_success,
            "actual_utility": state.actual_normalized_utility,
            "actual_net_gain": state.actual_net_gain,
            "direct_probability": scores["direct_llm"][split][index],
            "static_reliability_probability": scores["static_reliability"][split][index],
            "conformal_lower_gain": scores["factorized_conformal_lower_gain"][split][index],
            "factorized": prediction.to_json(),
        }
        for index, (state, prediction) in enumerate(zip(states, predictions))
    ]


def _static_records(trajectories):
    output = []
    for trajectory in trajectories:
        base = next(row for row in trajectory["stages"] if row["stage"] == "base")
        payload = dict(base["visible_observation"]["record"])
        hidden = trajectory["hidden_outcome_for_evaluation_only"]
        payload["label"] = int(hidden["success_label"])
        payload["utility"] = float(hidden["normalized_utility"])
        output.append(ActionRecord.from_json(payload))
    return output


def _fit_temperature(probabilities, labels):
    best_temperature = 1.0
    best_loss = float("inf")
    for step in range(2, 201):
        temperature = step / 20.0
        values = [_temperature_scale(value, temperature) for value in probabilities]
        loss = -_mean(
            label * math.log(_clip(probability))
            + (1 - label) * math.log(_clip(1.0 - probability))
            for label, probability in zip(labels, values)
        )
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _temperature_scale(probability, temperature):
    return _sigmoid(_logit(probability) / temperature)


def _group_key(state: FactorizedGainState):
    return f"{state.benchmark}::{state.group_id}"


def _worthy(state):
    return int(state.actual_net_gain > 0.0)


def _correlation(left, right):
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a-left_mean)*(b-right_mean) for a,b in zip(left,right))
    denominator = math.sqrt(sum((a-left_mean)**2 for a in left)*sum((b-right_mean)**2 for b in right))
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
