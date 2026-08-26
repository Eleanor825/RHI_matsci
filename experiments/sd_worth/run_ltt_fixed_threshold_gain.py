from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from harness_matsci.conformal_gain import ConformalGainObservation, fit_split_conformal_gain
from harness_matsci.factorized_gain_harness import (
    fit_factorized_gain_harness,
    fit_factorized_temperature,
)
from harness_matsci.learn_then_test_gain import (
    evaluate_fixed_threshold_gain_policy,
    fit_learn_then_test_gain_policy,
)
from harness_matsci.llm_worthiness import STRICT_GAIN_PROMPT_VERSION
from harness_matsci.sequential_safe_evidence import (
    build_factorized_gain_states,
    load_worthiness_prediction_caches,
)

from run_sequential_factorized_gain import SOURCE_CONTRACTS, _group_key, _load_jsonl
from run_strict_budgeted_gain import _select_gain_configuration


ALPHA = 0.10
DELTA = 0.05
BUDGET_FRACTION = 0.10


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
    parser.add_argument("--seed", type=int, default=4729)
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
    predictions = load_worthiness_prediction_caches(
        trajectories,
        model_cache_paths={
            model: Path(args.cache_dir) / f"{model}.json" for model in models
        },
        prompt_version=STRICT_GAIN_PROMPT_VERSION,
    )
    report = run_experiment(
        trajectories,
        predictions,
        prior_ids=prior_ids,
        cv_folds=args.cv_folds,
        seed=args.seed,
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
    evaluate_test,
):
    splits = {
        split: [row for row in trajectories if row["split"] == split]
        for split in ("train", "feedback", "acceptance", "test")
    }
    evaluation = {
        split: [
            row
            for row in splits[split]
            if not prior_ids or row["record_id"] not in prior_ids
        ]
        for split in ("acceptance", "test")
    }
    required = ("train", "feedback", "acceptance") + (("test",) if evaluate_test else ())
    if any(not splits[name] for name in required):
        raise ValueError("all required data roles must be non-empty")
    states_by_contract = {
        contract: {
            "train": build_factorized_gain_states(
                splits["train"],
                llm_predictions,
                source_stages=sources,
                gain_target="strict_realized",
            ),
            "feedback": build_factorized_gain_states(
                splits["feedback"],
                llm_predictions,
                source_stages=sources,
                gain_target="strict_realized",
            ),
            "acceptance": build_factorized_gain_states(
                evaluation["acceptance"],
                llm_predictions,
                source_stages=sources,
                gain_target="strict_realized",
            ),
            **(
                {
                    "test": build_factorized_gain_states(
                        evaluation["test"],
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
    probability_feedback, conformal_feedback, policy_feedback = _split_feedback_three(
        states["feedback"],
        seed,
    )
    base_model = fit_factorized_gain_harness(
        states["train"],
        blend_weight=chosen["blend_weight"],
        regularization=chosen["regularization"],
    )
    model = fit_factorized_temperature(base_model, probability_feedback)
    predictions = {name: model.predict(rows) for name, rows in states.items()}
    conformal = fit_split_conformal_gain(
        [
            ConformalGainObservation(
                state.record_id,
                prediction.expected_net_gain,
                _prediction_scale(prediction),
                state.actual_net_gain,
            )
            for state, prediction in zip(
                conformal_feedback,
                model.predict(conformal_feedback),
            )
        ],
        alpha=ALPHA,
    )
    lower_scores = {
        name: [
            conformal.predict_one(
                record_id=state.record_id,
                predicted_mean_gain=prediction.expected_net_gain,
                predicted_scale=_prediction_scale(prediction),
            ).lower_gain_bound
            for state, prediction in zip(rows, predictions[name])
        ]
        for name, rows in states.items()
    }
    policy_feedback_predictions = model.predict(policy_feedback)
    policy_feedback_scores = [
        conformal.predict_one(
            record_id=state.record_id,
            predicted_mean_gain=prediction.expected_net_gain,
            predicted_scale=_prediction_scale(prediction),
        ).lower_gain_bound
        for state, prediction in zip(policy_feedback, policy_feedback_predictions)
    ]
    thresholds = _threshold_family(lower_scores["train"])
    policy, candidate_evidence = fit_learn_then_test_gain_policy(
        candidate_thresholds=thresholds,
        feedback_scores=policy_feedback_scores,
        feedback_gains=[state.actual_net_gain for state in policy_feedback],
        feedback_record_ids=[state.record_id for state in policy_feedback],
        alpha=ALPHA,
        delta=DELTA,
        budget_fraction=BUDGET_FRACTION,
        thinning_seed=seed,
    )
    acceptance = None
    test = None
    if policy is not None:
        acceptance = evaluate_fixed_threshold_gain_policy(
            policy,
            lower_scores["acceptance"],
            [state.actual_net_gain for state in states["acceptance"]],
            [state.record_id for state in states["acceptance"]],
            alpha=ALPHA,
            delta=DELTA,
        )
        if evaluate_test:
            test = evaluate_fixed_threshold_gain_policy(
                policy,
                lower_scores["test"],
                [state.actual_net_gain for state in states["test"]],
                [state.record_id for state in states["test"]],
                alpha=ALPHA,
                delta=DELTA,
            )
    return {
        "schema_version": "ltt-fixed-threshold-strict-gain-v1",
        "claim_boundary": {
            "route_output": False,
            "policy_type": "fixed action-wise threshold with outcome-independent thinning",
            "risk_assumption": "i.i.d. deployment actions from the frozen task mixture",
        },
        "sizes": {
            **{name: len(rows) for name, rows in states.items()},
            "probability_feedback": len(probability_feedback),
            "conformal_feedback": len(conformal_feedback),
            "policy_feedback": len(policy_feedback),
        },
        "selection": selection,
        "model": model.to_json(),
        "conformal": conformal.to_json(),
        "threshold_family": thresholds,
        "candidate_evidence": [row.to_json() for row in candidate_evidence],
        "policy": policy.to_json() if policy is not None else None,
        "summary": {
            "feedback_policy_certified": policy is not None,
            "acceptance": acceptance,
            "test": test,
        },
    }


def _split_feedback_three(states, seed):
    groups = sorted({_group_key(state) for state in states})
    partitions = [set(), set(), set()]
    for group in groups:
        digest = hashlib.sha256(f"h4-feedback::{seed}::{group}".encode()).hexdigest()
        partitions[int(digest[:16], 16) % 3].add(group)
    if any(not partition for partition in partitions):
        partitions = [set(groups[index::3]) for index in range(3)]
    rows = [
        [state for state in states if _group_key(state) in partition]
        for partition in partitions
    ]
    if any(not partition_rows for partition_rows in rows):
        raise ValueError("three-way feedback split produced an empty partition")
    return tuple(rows)


def _threshold_family(scores):
    ordered = sorted(float(score) for score in scores)
    quantiles = (0.50, 0.70, 0.80, 0.90, 0.95)
    thresholds = {0.0}
    for quantile in quantiles:
        index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
        thresholds.add(max(0.0, ordered[index]))
    return sorted(thresholds)


def _prediction_scale(prediction):
    return math.sqrt(max(1e-12, prediction.total_gain_variance))


if __name__ == "__main__":
    main()
