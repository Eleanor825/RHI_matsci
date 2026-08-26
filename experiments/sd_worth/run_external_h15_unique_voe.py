from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from harness_matsci.evidence_value import (
    EvidenceValueObservation,
    cross_fitted_evidence_value_predictions,
    evidence_value_observation,
    fit_evidence_value_model,
)
from harness_matsci.external_h15_unique import TASK
from harness_matsci.external_h15_unique_tools import (
    ExternalH15UniqueEvidenceEnvironment,
)
from harness_matsci.schema import ActionRecord
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    EstimatedGainProjector,
    GainConfig,
    gain_targets,
)
from harness_matsci.source_state import (
    build_source_observations,
    cross_fitted_source_state_predictions,
    fit_source_state_model,
)
from harness_matsci.two_stage_policy import (
    evaluate_two_stage_policy,
    select_execution_threshold,
    select_two_stage_policy,
)
from harness_matsci.tail_voi import (
    cross_fitted_tail_voi_predictions,
    fit_tail_voi_model,
)


BASE_SOURCES = ("composition", "structure")
TOOL_SOURCES = ("composition", "structure", "high_fidelity")
TOOL_COST = 0.020
COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--exclude-tail-methods", action="store_true")
    args = parser.parse_args()
    result = run_unique_voe(
        args.fold_dir,
        args.source,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
        alpha=args.alpha,
        delta=args.delta,
        development_only=not args.confirmatory,
        include_tail_methods=not args.exclude_tail_methods,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


def run_unique_voe(
    fold_dir,
    source,
    *,
    seed,
    crossfit_folds,
    alpha=0.20,
    delta=0.05,
    development_only=True,
    include_tail_methods=True,
):
    return run_single_task_voe(
        fold_dir,
        source,
        task=TASK,
        environment_class=ExternalH15UniqueEvidenceEnvironment,
        experiment_name="h15_powered_unique_two_stage_voe",
        tool_decision_mode="calibrated_hurdle",
        seed=seed,
        crossfit_folds=crossfit_folds,
        alpha=alpha,
        delta=delta,
        development_only=development_only,
        include_tail_methods=include_tail_methods,
    )


def run_single_task_voe(
    fold_dir,
    source,
    *,
    task,
    environment_class,
    experiment_name,
    tool_decision_mode="calibrated_hurdle",
    evidence_value_target_mode="zero_threshold",
    evidence_value_model_family="extra_trees",
    base_sources=BASE_SOURCES,
    tool_sources=TOOL_SOURCES,
    base_execution_coverages=None,
    tool_execution_coverages=None,
    include_record_predictions=False,
    development_only=True,
    include_tail_methods=True,
    alpha=0.20,
    delta=0.05,
    policy_selection_mode="absolute_threshold",
    use_decision_instability=False,
    seed,
    crossfit_folds,
):
    root = Path(fold_dir)
    splits = {
        split: _load_records(root / f"{task}.{split}.jsonl")
        for split in ("train", "feedback", "acceptance", "test")
    }
    config = GainConfig(utility_mode="continuous_realized")
    normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
    reservations = BudgetReservationValues.fit(
        splits["train"],
        normalizer,
        config,
        budget_fraction=0.10,
        strategy="potential_success_slot_value",
    )
    projector = EstimatedGainProjector(normalizer, reservations, config)
    targets = {
        split: gain_targets(records, normalizer, config, reservations)
        for split, records in splits.items()
    }
    target_by_id = {
        target.record_id: target
        for rows in targets.values()
        for target in rows
    }
    environment = environment_class.fit(
        source,
        splits["train"],
        seed=seed,
    )
    raw = {
        state: {
            split: build_source_observations(
                environment,
                records,
                target_by_id,
                source_names=sources,
                gain_projector=projector,
            )
            for split, records in splits.items()
            if split != "train"
        }
        for state, sources in (("base", base_sources), ("tool", tool_sources))
    }
    feedback_base = cross_fitted_source_state_predictions(
        raw["base"]["feedback"],
        folds=crossfit_folds,
        seed=seed + 11,
        calibrate_source_margins=False,
        use_decision_instability=use_decision_instability,
    )
    feedback_tool = cross_fitted_source_state_predictions(
        raw["tool"]["feedback"],
        folds=crossfit_folds,
        seed=seed + 29,
        calibrate_source_margins=False,
        use_decision_instability=use_decision_instability,
    )
    if tool_decision_mode == "direct_high_fidelity_gain":
        feedback_tool = _use_direct_high_fidelity_gain(
            feedback_tool,
            raw["tool"]["feedback"],
        )
    elif tool_decision_mode != "calibrated_hurdle":
        raise ValueError(f"unknown tool decision mode {tool_decision_mode}")
    if evidence_value_target_mode == "zero_threshold":
        feedback_voe = [
            evidence_value_observation(
                base,
                tool,
                realized_gain=target_by_id[base.record_id].gain,
                tool_cost=TOOL_COST,
                source_observation=source_observation,
            )
            for base, tool, source_observation in zip(
                feedback_base,
                feedback_tool,
                raw["base"]["feedback"],
            )
        ]
        provisional_thresholds = []
    elif evidence_value_target_mode == "cross_fitted_policy_conditioned":
        feedback_voe, provisional_thresholds = _cross_fitted_policy_conditioned_voe(
            feedback_base,
            feedback_tool,
            raw["base"]["feedback"],
            [target.worthy for target in targets["feedback"]],
            [target.gain for target in targets["feedback"]],
            tool_cost=TOOL_COST,
            coverages=COVERAGES,
            base_execution_coverages=base_execution_coverages,
            tool_execution_coverages=tool_execution_coverages,
            max_empirical_risk=alpha,
            folds=crossfit_folds,
            seed=seed + 67,
        )
    else:
        raise ValueError(
            f"unknown evidence-value target mode {evidence_value_target_mode}"
        )
    feedback_voe_no_source = [
        _zero_source_features(row) for row in feedback_voe
    ]
    feedback_acquisition = {
        "source": cross_fitted_evidence_value_predictions(
            feedback_voe,
            seed=seed + 101,
            folds=crossfit_folds,
            replicas=3,
            trees=80,
            model_family=evidence_value_model_family,
        ),
        "no_source": cross_fitted_evidence_value_predictions(
            feedback_voe_no_source,
            seed=seed + 211,
            folds=crossfit_folds,
            replicas=3,
            trees=80,
            model_family=evidence_value_model_family,
        ),
    }
    feedback_tail = None
    if include_tail_methods:
        feedback_tail = {
            "source": cross_fitted_tail_voi_predictions(
                feedback_voe,
                seed=seed + 251,
                folds=crossfit_folds,
                replicas=3,
            ),
            "no_source": cross_fitted_tail_voi_predictions(
                feedback_voe_no_source,
                seed=seed + 271,
                folds=crossfit_folds,
                replicas=3,
            ),
        }
    base_model = fit_source_state_model(
        raw["base"]["feedback"],
        calibrate_source_margins=False,
        use_decision_instability=use_decision_instability,
    )
    tool_model = fit_source_state_model(
        raw["tool"]["feedback"],
        calibrate_source_margins=False,
        use_decision_instability=use_decision_instability,
    )
    source_model = fit_evidence_value_model(
        feedback_voe,
        seed=seed + 307,
        replicas=5,
        trees=160,
        model_family=evidence_value_model_family,
    )
    no_source_model = fit_evidence_value_model(
        feedback_voe_no_source,
        seed=seed + 401,
        replicas=5,
        trees=160,
        model_family=evidence_value_model_family,
    )
    source_tail_model = None
    no_source_tail_model = None
    if include_tail_methods:
        source_tail_model = fit_tail_voi_model(
            feedback_voe,
            seed=seed + 431,
            replicas=5,
        )
        no_source_tail_model = fit_tail_voi_model(
            feedback_voe_no_source,
            seed=seed + 461,
            replicas=5,
        )
    state = {"feedback": {"base": feedback_base, "tool": feedback_tool}}
    scores = {
        "feedback": {
            "source_voi_expected": [row.expected_value for row in feedback_acquisition["source"]],
            "source_voi_probability": [row.positive_value_probability for row in feedback_acquisition["source"]],
            "no_source_voi_expected": [row.expected_value for row in feedback_acquisition["no_source"]],
            "no_source_voi_probability": [row.positive_value_probability for row in feedback_acquisition["no_source"]],
            "external_variance": [row.calibrated_external_variance for row in feedback_base],
        }
    }
    if include_tail_methods:
        scores["feedback"].update(
            {
                "source_voi_tail": [
                    row.tail_score for row in feedback_tail["source"]
                ],
                "no_source_voi_tail": [
                    row.tail_score for row in feedback_tail["no_source"]
                ],
            }
        )
    for split in ("acceptance", "test"):
        base = base_model.predict(raw["base"][split])
        tool = tool_model.predict(raw["tool"][split])
        if tool_decision_mode == "direct_high_fidelity_gain":
            tool = _use_direct_high_fidelity_gain(tool, raw["tool"][split])
        state[split] = {"base": base, "tool": tool}
        observations = [
            _observation(row, source_observation)
            for row, source_observation in zip(base, raw["base"][split])
        ]
        no_source_observations = [
            _zero_source_features(row) for row in observations
        ]
        source_predictions = source_model.predict(observations)
        no_source_predictions = no_source_model.predict(no_source_observations)
        scores[split] = {
            "source_voi_expected": [row.expected_value for row in source_predictions],
            "source_voi_probability": [row.positive_value_probability for row in source_predictions],
            "no_source_voi_expected": [row.expected_value for row in no_source_predictions],
            "no_source_voi_probability": [row.positive_value_probability for row in no_source_predictions],
            "external_variance": [row.calibrated_external_variance for row in base],
        }
        if include_tail_methods:
            source_tail_predictions = source_tail_model.predict(observations)
            no_source_tail_predictions = no_source_tail_model.predict(
                no_source_observations
            )
            scores[split].update(
                {
                    "source_voi_tail": [
                        row.tail_score for row in source_tail_predictions
                    ],
                    "no_source_voi_tail": [
                        row.tail_score for row in no_source_tail_predictions
                    ],
                }
            )
    labels = {
        split: [target.worthy for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    gains = {
        split: [target.gain for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    policies = {}
    for method in scores["feedback"]:
        policy = select_two_stage_policy(
            scores["feedback"][method],
            state["feedback"]["base"],
            state["feedback"]["tool"],
            labels["feedback"],
            gains["feedback"],
            tool_cost=TOOL_COST,
            coverages=COVERAGES,
            base_execution_coverages=base_execution_coverages,
            tool_execution_coverages=tool_execution_coverages,
            max_empirical_risk=alpha,
            selection_mode=policy_selection_mode,
        )
        acceptance = evaluate_two_stage_policy(
            policy,
            scores["acceptance"][method],
            state["acceptance"]["base"],
            state["acceptance"]["tool"],
            labels["acceptance"],
            gains["acceptance"],
            tool_cost=TOOL_COST,
            alpha=alpha,
            delta=delta,
        )
        test = evaluate_two_stage_policy(
            policy,
            scores["test"][method],
            state["test"]["base"],
            state["test"]["tool"],
            labels["test"],
            gains["test"],
            tool_cost=TOOL_COST,
            alpha=alpha,
            delta=delta,
        )
        policies[method] = {
            "feedback_policy": policy.to_json(),
            "acceptance_certificate": acceptance.to_json(),
            "test_evaluation": test.to_json(),
            "deployed": acceptance.certified,
            "deployed_test_population_gain": (
                test.population_gain_after_cost if acceptance.certified else 0.0
            ),
        }
    result = {
        "experiment": experiment_name,
        "development_only": development_only,
        "include_tail_methods": include_tail_methods,
        "alpha": alpha,
        "delta": delta,
        "task": task,
        "crossfit_folds": crossfit_folds,
        "tool_cost": TOOL_COST,
        "tool_decision_mode": tool_decision_mode,
        "evidence_value_target_mode": evidence_value_target_mode,
        "evidence_value_model_family": evidence_value_model_family,
        "base_sources": list(base_sources),
        "tool_sources": list(tool_sources),
        "base_execution_coverages": (
            list(base_execution_coverages)
            if base_execution_coverages is not None
            else list(COVERAGES)
        ),
        "tool_execution_coverages": (
            list(tool_execution_coverages)
            if tool_execution_coverages is not None
            else list(COVERAGES)
        ),
        "policy_selection_mode": policy_selection_mode,
        "use_decision_instability": use_decision_instability,
        "cross_fitted_provisional_execution_thresholds": provisional_thresholds,
        "gain_projector": projector.to_json(),
        "feedback_realized_voe": {
            "count": len(feedback_voe),
            "mean": sum(row.realized_value for row in feedback_voe) / len(feedback_voe),
            "positive_fraction": sum(row.realized_value > 0.0 for row in feedback_voe) / len(feedback_voe),
        },
        "policies": policies,
    }
    if include_record_predictions:
        result["record_predictions"] = {
            split: [
                {
                    "record_id": record.record_id,
                    "label": int(label),
                    "gain": float(gain),
                    "base_p_positive_gain": float(base.p_positive_gain),
                    "base_tail_risk_score": float(base.tail_risk_score),
                    "base_internal_variance": float(base.calibrated_internal_variance),
                    "base_external_variance": float(base.calibrated_external_variance),
                    "tool_tail_risk_score": float(tool.tail_risk_score),
                    "acquisition_scores": {
                        method: float(scores[split][method][index])
                        for method in scores[split]
                    },
                }
                for index, (record, label, gain, base, tool) in enumerate(
                    zip(
                        splits[split],
                        labels[split],
                        gains[split],
                        state[split]["base"],
                        state[split]["tool"],
                    )
                )
            ]
            for split in ("feedback", "acceptance", "test")
        }
    return result


def _observation(prediction, source_observation):
    from harness_matsci.evidence_value import source_disagreement_features

    return EvidenceValueObservation(
        record_id=prediction.record_id,
        benchmark=prediction.benchmark,
        base_mean_gain=prediction.mean_gain,
        base_p_positive_gain=prediction.p_positive_gain,
        base_expected_shortfall=prediction.expected_shortfall,
        base_tail_risk_score=prediction.tail_risk_score,
        base_internal_variance=prediction.calibrated_internal_variance,
        base_external_variance=prediction.calibrated_external_variance,
        tool_cost=TOOL_COST,
        realized_value=0.0,
        **source_disagreement_features(source_observation),
    )


def _zero_source_features(observation):
    return replace(
        observation,
        base_external_variance=0.0,
        base_interaction_variance=0.0,
        base_source_range=0.0,
        base_source_variance=0.0,
        base_source_sign_disagreement=0.0,
        base_source_positive_fraction=0.0,
        base_external_mean=0.0,
        base_internal_external_gap=0.0,
        base_external_consensus_variance=0.0,
        base_source_interaction_variance=0.0,
        context_evidence_support=0.0,
        context_evidence_conflict=0.0,
        context_ood_score=0.0,
        context_source_reliability=0.0,
    )


def _use_direct_high_fidelity_gain(predictions, observations):
    high_fidelity_index = observations[0].source_names.index("high_fidelity")
    return [
        replace(
            prediction,
            mean_gain=float(observation.predicted_gain[1][high_fidelity_index]),
            tail_risk_score=float(observation.predicted_gain[1][high_fidelity_index]),
        )
        for prediction, observation in zip(predictions, observations)
    ]


def _cross_fitted_policy_conditioned_voe(
    base_predictions,
    tool_predictions,
    source_observations,
    labels,
    gains,
    *,
    tool_cost,
    coverages,
    base_execution_coverages=None,
    tool_execution_coverages=None,
    max_empirical_risk,
    folds,
    seed,
):
    assignments = [
        _stable_fold(prediction.record_id, prediction.benchmark, folds=folds, seed=seed)
        for prediction in base_predictions
    ]
    output = [None] * len(base_predictions)
    thresholds = []
    for fold in range(folds):
        train_indices = [
            index for index, assignment in enumerate(assignments) if assignment != fold
        ]
        held_indices = [
            index for index, assignment in enumerate(assignments) if assignment == fold
        ]
        if not held_indices:
            continue
        base_threshold = select_execution_threshold(
            [base_predictions[index] for index in train_indices],
            [labels[index] for index in train_indices],
            [gains[index] for index in train_indices],
            coverages=base_execution_coverages or coverages,
            max_empirical_risk=max_empirical_risk,
        )
        tool_threshold = select_execution_threshold(
            [tool_predictions[index] for index in train_indices],
            [labels[index] for index in train_indices],
            [gains[index] for index in train_indices],
            coverages=tool_execution_coverages or coverages,
            max_empirical_risk=max_empirical_risk,
        )
        thresholds.append(
            {
                "fold": fold,
                "base": base_threshold.to_json(),
                "tool": tool_threshold.to_json(),
                "train_count": len(train_indices),
                "held_count": len(held_indices),
            }
        )
        for index in held_indices:
            output[index] = evidence_value_observation(
                base_predictions[index],
                tool_predictions[index],
                realized_gain=gains[index],
                tool_cost=tool_cost,
                source_observation=source_observations[index],
                base_execution_threshold=base_threshold.threshold,
                tool_execution_threshold=tool_threshold.threshold,
            )
    if any(row is None for row in output):
        raise RuntimeError("policy-conditioned cross-fitting missed feedback records")
    return [row for row in output if row is not None], thresholds


def _stable_fold(record_id, benchmark, *, folds, seed):
    digest = hashlib.sha256(f"{seed}:{benchmark}:{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _load_records(path):
    return [
        ActionRecord.from_json(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line
    ]


def _markdown(result):
    status = (
        "Development result; all feedback state and VoE scores are cross-fitted."
        if result["development_only"]
        else "Confirmatory result; all feedback state and VoE scores are cross-fitted."
    )
    lines = [
        f"# {result['experiment']}",
        "",
        status,
        "",
        "| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for method, values in result["policies"].items():
        policy = values["feedback_policy"]
        acceptance = values["acceptance_certificate"]
        test = values["test_evaluation"]
        lines.append(
            f"| `{method}` | {policy['target_acquisition_coverage']:.4f} | "
            f"{acceptance['acquired_count']} | {acceptance['executed_count']} | "
            f"{acceptance['selective_risk']:.4f} | {acceptance['risk_upper_bound']:.4f} | "
            f"`{values['deployed']}` | {test['acquired_count']} | "
            f"{test['executed_count']} | {test['selective_risk']:.4f} | "
            f"{values['deployed_test_population_gain']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Feedback realized VoE mean: `{result['feedback_realized_voe']['mean']:+.6f}`; ",
            f"positive fraction: `{result['feedback_realized_voe']['positive_fraction']:.4f}`.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
