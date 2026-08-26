from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from harness_matsci.budgeted_gain_policy import (
    BudgetedGainPolicy,
    certify_budgeted_gain_policy,
    evaluate_budgeted_gain_policy,
)
from harness_matsci.conformal_gain import ConformalGainObservation, fit_split_conformal_gain
from harness_matsci.factorized_gain_harness import (
    fit_factorized_gain_harness,
    fit_factorized_temperature,
)
from harness_matsci.metrics import binary_metrics
from harness_matsci.llm_worthiness import (
    PROMPT_VERSION,
    STRICT_GAIN_PROMPT_VERSION,
)
from harness_matsci.schema import ActionRecord
from harness_matsci.sequential_safe_evidence import (
    build_factorized_gain_states,
    load_worthiness_prediction_caches,
)
from harness_matsci.training import train_gate_with_features

from run_sequential_factorized_gain import (
    BLEND_WEIGHTS,
    REGULARIZATIONS,
    SOURCE_CONTRACTS,
    STATIC_RELIABILITY_FEATURES,
    _fit_temperature,
    _group_folds,
    _group_key,
    _load_jsonl,
    _mean,
    _split_feedback,
    _static_records,
    _temperature_scale,
)


BUDGET_FRACTION = 0.10
RISK_ALPHAS = (0.10, 0.15, 0.20)


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
    parser.add_argument("--seed", type=int, default=3729)
    parser.add_argument(
        "--cache-prompt-version",
        choices=(PROMPT_VERSION, STRICT_GAIN_PROMPT_VERSION),
        default=STRICT_GAIN_PROMPT_VERSION,
    )
    parser.add_argument("--acceptance-only", action="store_true")
    args = parser.parse_args()

    trajectories = _load_jsonl(Path(args.trajectories))
    if args.acceptance_only:
        trajectories = [row for row in trajectories if row["split"] != "test"]
    prior_ids = set()
    if args.fresh_relative_to:
        prior_ids = {
            row["record_id"] for row in _load_jsonl(Path(args.fresh_relative_to))
        }
    models = tuple(value for value in args.models.split(",") if value)
    llm_predictions = load_worthiness_prediction_caches(
        trajectories,
        model_cache_paths={
            model: Path(args.cache_dir) / f"{model}.json" for model in models
        },
        prompt_version=args.cache_prompt_version,
    )
    report = run_experiment(
        trajectories,
        llm_predictions,
        prior_ids=prior_ids,
        cv_folds=args.cv_folds,
        seed=args.seed,
        cache_prompt_version=args.cache_prompt_version,
        evaluate_test=not args.acceptance_only,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))


def run_experiment(
    trajectories,
    llm_predictions,
    *,
    prior_ids,
    cv_folds,
    seed,
    cache_prompt_version,
    evaluate_test=True,
):
    split_rows = {
        split: [row for row in trajectories if row["split"] == split]
        for split in ("train", "feedback", "acceptance", "test")
    }
    evaluation_rows = {
        split: [
            row
            for row in split_rows[split]
            if not prior_ids or row["record_id"] not in prior_ids
        ]
        for split in ("acceptance", "test")
    }
    required_splits = ("train", "feedback", "acceptance") + (
        ("test",) if evaluate_test else ()
    )
    if any(not split_rows[split] for split in required_splits) or any(
        not evaluation_rows[split]
        for split in ("acceptance",) + (("test",) if evaluate_test else ())
    ):
        raise ValueError("all data roles must be non-empty")

    states_by_contract = {
        contract: {
            "train": build_factorized_gain_states(
                split_rows["train"],
                llm_predictions,
                source_stages=sources,
                gain_target="strict_realized",
            ),
            "feedback": build_factorized_gain_states(
                split_rows["feedback"],
                llm_predictions,
                source_stages=sources,
                gain_target="strict_realized",
            ),
            "acceptance": build_factorized_gain_states(
                evaluation_rows["acceptance"],
                llm_predictions,
                source_stages=sources,
                gain_target="strict_realized",
            ),
            **(
                {
                    "test": build_factorized_gain_states(
                        evaluation_rows["test"],
                        llm_predictions,
                        source_stages=sources,
                        gain_target="strict_realized",
                    )
                }
                if evaluate_test
                else {}
            ),
        }
        for contract, sources in SOURCE_CONTRACTS.items()
    }
    selection = _select_gain_configuration(
        states_by_contract,
        cv_folds=cv_folds,
        seed=seed,
    )
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

    static_scores, static_metadata = _static_reliability_scores(
        split_rows,
        evaluation_rows,
        states,
        probability_feedback,
    )
    direct_temperature = _fit_temperature(
        [_mean(state.model_worthiness_probabilities) for state in probability_feedback],
        [_worthy(state) for state in probability_feedback],
    )
    direct_judge = {
        split: [
            _temperature_scale(
                _mean(state.model_worthiness_probabilities),
                direct_temperature,
            )
            for state in rows
        ]
        for split, rows in states.items()
    }
    component_gain = {
        split: [_direct_component_gain(state) for state in rows]
        for split, rows in states.items()
    }
    scores = {
        "factorized_conformal_lower_gain": lower_gain,
        "factorized_expected_gain": {
            split: [prediction.expected_net_gain for prediction in rows]
            for split, rows in predictions.items()
        },
        "llm_component_gain": component_gain,
        "direct_llm_judge": direct_judge,
        "static_reliability": static_scores,
    }
    policies = {
        "factorized_conformal_lower_gain": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.0
        ),
        "factorized_expected_gain": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.0
        ),
        "llm_component_gain": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.0
        ),
        "direct_llm_judge": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.5
        ),
        "static_reliability": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.5
        ),
    }
    certificates = {
        method: {
            f"alpha_{alpha:.2f}": certify_budgeted_gain_policy(
                policies[method],
                values["acceptance"],
                [state.actual_net_gain for state in states["acceptance"]],
                alpha=alpha,
            )
            for alpha in RISK_ALPHAS
        }
        for method, values in scores.items()
    }
    test_evaluations = (
        {
            method: evaluate_budgeted_gain_policy(
                policies[method],
                values["test"],
                [state.actual_net_gain for state in states["test"]],
            )
            for method, values in scores.items()
        }
        if evaluate_test
        else None
    )

    return {
        "schema_version": "strict-budgeted-factorized-gain-harness-v1",
        "llm_prompt_version": cache_prompt_version,
        "claim_boundary": {
            "realized_gain": (
                "success-gated normalized scientific utility minus action cost "
                "and irreversible failure harm"
            ),
            "reservation_value": (
                "excluded from realized outcomes; resource scarcity is enforced "
                "by a global at-most budget"
            ),
            "route_output": False,
        },
        "prior_record_count_excluded_from_acceptance_test": len(prior_ids),
        "evaluation_phase": "full" if evaluate_test else "acceptance_only",
        "sizes": {split: len(rows) for split, rows in states.items()},
        "selection": selection,
        "model": model.to_json(),
        "conformal": conformal.to_json(),
        "direct_llm_judge": {
            "temperature": direct_temperature,
            "input": "mean p_positive_gain across model replicas",
        },
        "static_reliability": static_metadata,
        "policies": {name: policy.to_json() for name, policy in policies.items()},
        "summary": {
            "probability_metrics": (
                {
                    "factorized_worthiness": binary_metrics(
                        [_worthy(state) for state in states["test"]],
                        [row.action_worthiness for row in predictions["test"]],
                        threshold=0.5,
                    ),
                    "static_reliability": binary_metrics(
                        [_worthy(state) for state in states["test"]],
                        static_scores["test"],
                        threshold=0.5,
                    ),
                    "direct_llm_judge": binary_metrics(
                        [_worthy(state) for state in states["test"]],
                        direct_judge["test"],
                        threshold=0.5,
                    ),
                }
                if evaluate_test
                else None
            ),
            "gain_metrics": (
                _gain_metrics(states["test"], predictions["test"])
                if evaluate_test
                else None
            ),
            "conformal_coverage": (
                _conformal_coverage(states["test"], predictions["test"], conformal)
                if evaluate_test
                else None
            ),
            "acceptance_certificates": certificates,
            "test_budgeted_evaluations": test_evaluations,
        },
        "acceptance_predictions": _prediction_rows(
            states["acceptance"], predictions["acceptance"], scores, "acceptance"
        ),
        "test_predictions": (
            _prediction_rows(states["test"], predictions["test"], scores, "test")
            if evaluate_test
            else []
        ),
    }


def _select_gain_configuration(states_by_contract, *, cv_folds, seed):
    policy = BudgetedGainPolicy(BUDGET_FRACTION, minimum_score=0.0)
    candidates = []
    for contract, splits in states_by_contract.items():
        states = splits["train"]
        folds = _group_folds(states, cv_folds, seed)
        for blend_weight in BLEND_WEIGHTS:
            for regularization in REGULARIZATIONS:
                predicted = []
                actual = []
                labels = []
                probabilities = []
                for held_groups in folds:
                    fit_rows = [
                        state for state in states if _group_key(state) not in held_groups
                    ]
                    held_rows = [
                        state for state in states if _group_key(state) in held_groups
                    ]
                    model = fit_factorized_gain_harness(
                        fit_rows,
                        blend_weight=blend_weight,
                        regularization=regularization,
                    )
                    rows = model.predict(held_rows)
                    predicted.extend(row.expected_net_gain for row in rows)
                    actual.extend(state.actual_net_gain for state in held_rows)
                    labels.extend(_worthy(state) for state in held_rows)
                    probabilities.extend(row.action_worthiness for row in rows)
                utility = evaluate_budgeted_gain_policy(policy, predicted, actual)
                metrics = binary_metrics(labels, probabilities, threshold=0.5)
                gain_mae = _mean(abs(left - right) for left, right in zip(predicted, actual))
                candidates.append(
                    {
                        "source_contract": contract,
                        "blend_weight": blend_weight,
                        "regularization": regularization,
                        "cross_validated_budgeted_utility": utility,
                        "gain_mae": gain_mae,
                        "metrics": metrics,
                    }
                )
    selected = min(
        candidates,
        key=lambda row: (
            -row["cross_validated_budgeted_utility"]["population_net_gain"],
            row["cross_validated_budgeted_utility"]["selective_risk"],
            row["gain_mae"],
            row["metrics"]["log_loss"],
            row["source_contract"],
            row["blend_weight"],
            row["regularization"],
        ),
    )
    return {
        "criterion": (
            "maximize train-group-held-out strict net scientific gain under a "
            "global at-most 10% budget"
        ),
        "selected": selected,
        "candidates": candidates,
    }


def _static_reliability_scores(
    split_rows,
    evaluation_rows,
    states,
    probability_feedback,
):
    records = {
        "train": _static_records(split_rows["train"]),
        "feedback": _static_records(split_rows["feedback"]),
        "acceptance": _static_records(evaluation_rows["acceptance"]),
        **(
            {"test": _static_records(evaluation_rows["test"])}
            if evaluation_rows["test"]
            else {}
        ),
    }
    labels_by_id = {
        state.record_id: _worthy(state)
        for split in states.values()
        for state in split
    }
    for split in records:
        records[split] = [
            ActionRecord(
                **{
                    **record.to_json(),
                    "label": labels_by_id[record.record_id],
                }
            )
            for record in records[split]
        ]
    feature_names = [
        name
        for name in STATIC_RELIABILITY_FEATURES
        if any(name in record.features for record in records["train"])
    ]
    feedback_ids = {state.record_id for state in probability_feedback}
    feedback_subset = [
        record for record in records["feedback"] if record.record_id in feedback_ids
    ]
    gate = train_gate_with_features(
        records["train"],
        feedback_subset,
        feature_names=feature_names,
        epochs=500,
        learning_rate=0.08,
        l2=0.01,
        balance_benchmarks=True,
    )
    temperature = _fit_temperature(
        gate.predict_proba(feedback_subset),
        [record.label for record in feedback_subset],
    )
    scores = {
        split: [
            _temperature_scale(value, temperature)
            for value in gate.predict_proba(rows)
        ]
        for split, rows in records.items()
    }
    return scores, {
        "temperature": temperature,
        "features": feature_names,
        "gate": gate.to_json(),
    }


def _direct_component_gain(state):
    gains = [
        probability * utility
        - 0.15 * state.normalized_cost
        - 0.25 * (1.0 - probability) * state.failure_harm
        for probability, utility in zip(
            state.model_success_probabilities,
            state.model_utility_means,
        )
    ]
    return _mean(gains)


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
    return {
        "target": 0.9,
        "empirical": _mean(float(value) for value in covered),
        "count": len(covered),
    }


def _prediction_rows(states, predictions, scores, split):
    return [
        {
            "record_id": state.record_id,
            "benchmark": state.benchmark,
            "actual_success": state.actual_success,
            "actual_normalized_utility": state.actual_normalized_utility,
            "actual_strict_net_gain": state.actual_net_gain,
            "worthy": _worthy(state),
            "factorized": prediction.to_json(),
            "scores": {method: values[split][index] for method, values in scores.items()},
        }
        for index, (state, prediction) in enumerate(zip(states, predictions))
    ]


def _worthy(state):
    return int(state.actual_net_gain > 0.0)


def _correlation(left, right):
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
