from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from harness_matsci.evidence_value import (
    cross_fitted_evidence_value_predictions,
    evidence_value_observation,
    fit_evidence_value_model,
)
from harness_matsci.external_h17_pairwise import TASK
from harness_matsci.external_h17_pairwise_tools import (
    ExternalH17PairwiseEvidenceEnvironment,
)
from harness_matsci.multi_tool_policy import (
    evaluate_multi_tool_policy,
    select_certified_tool_contract,
)
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

from run_external_h15_unique_voe import (
    _cross_fitted_policy_conditioned_voe,
    _load_records,
    _use_direct_high_fidelity_gain,
)


TOOLS = ("composition", "structure", "high_fidelity")
TOOL_COSTS = {"composition": 0.002, "structure": 0.002, "high_fidelity": 0.020}
COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1711)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--policy-selection-mode",
        choices=("absolute_threshold", "rank_budget"),
        default="absolute_threshold",
    )
    args = parser.parse_args()
    result = run_multitool_voe(
        args.fold_dir,
        args.source,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
        policy_selection_mode=args.policy_selection_mode,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


def run_multitool_voe(
    fold_dir,
    source,
    *,
    task=TASK,
    environment_class=ExternalH17PairwiseEvidenceEnvironment,
    experiment_name="h17_numeric_multitool_voe_development",
    development_only=True,
    seed,
    crossfit_folds,
    policy_selection_mode,
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
        source, splits["train"], seed=seed
    )
    source_sets = {
        "base": ("agent_prior",),
        **{tool: ("agent_prior", tool) for tool in TOOLS},
    }
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
        for state, sources in source_sets.items()
    }
    feedback_state = {
        state: cross_fitted_source_state_predictions(
            raw[state]["feedback"],
            folds=crossfit_folds,
            seed=seed + 31 * (index + 1),
            calibrate_source_margins=False,
        )
        for index, state in enumerate(source_sets)
    }
    feedback_state["high_fidelity"] = _use_direct_high_fidelity_gain(
        feedback_state["high_fidelity"], raw["high_fidelity"]["feedback"]
    )
    state_models = {
        state: fit_source_state_model(
            raw[state]["feedback"], calibrate_source_margins=False
        )
        for state in source_sets
    }
    state = {"feedback": feedback_state}
    for split in ("acceptance", "test"):
        state[split] = {
            name: model.predict(raw[name][split])
            for name, model in state_models.items()
        }
        state[split]["high_fidelity"] = _use_direct_high_fidelity_gain(
            state[split]["high_fidelity"], raw["high_fidelity"][split]
        )

    feedback_observations = {}
    provisional_thresholds = {}
    for tool in TOOLS:
        rows, thresholds = _cross_fitted_policy_conditioned_voe(
            state["feedback"]["base"],
            state["feedback"][tool],
            raw["base"]["feedback"],
            [target.worthy for target in targets["feedback"]],
            [target.gain for target in targets["feedback"]],
            tool_cost=TOOL_COSTS[tool],
            coverages=COVERAGES,
            base_execution_coverages=(0.0,),
            tool_execution_coverages=COVERAGES,
            max_empirical_risk=0.20,
            folds=crossfit_folds,
            seed=seed + 401 + 17 * TOOLS.index(tool),
        )
        feedback_observations[tool] = _augment_observations(
            rows, splits["feedback"], tool
        )
        provisional_thresholds[tool] = thresholds

    pooled_feedback = [
        observation
        for tool in TOOLS
        for observation in feedback_observations[tool]
    ]
    pooled_no_source = [_remove_external_context(row) for row in pooled_feedback]
    source_crossfit = cross_fitted_evidence_value_predictions(
        pooled_feedback,
        seed=seed + 701,
        folds=crossfit_folds,
        replicas=3,
        model_family="regularized_linear",
    )
    no_source_crossfit = cross_fitted_evidence_value_predictions(
        pooled_no_source,
        seed=seed + 709,
        folds=crossfit_folds,
        replicas=3,
        model_family="regularized_linear",
    )
    source_model = fit_evidence_value_model(
        pooled_feedback,
        seed=seed + 811,
        replicas=5,
        model_family="regularized_linear",
    )
    no_source_model = fit_evidence_value_model(
        pooled_no_source,
        seed=seed + 821,
        replicas=5,
        model_family="regularized_linear",
    )

    source_scores = {"feedback": _unpool_scores(source_crossfit)}
    no_source_scores = {"feedback": _unpool_scores(no_source_crossfit)}
    for split in ("acceptance", "test"):
        per_tool = {
            tool: _augment_observations(
                [
                    evidence_value_observation(
                        base,
                        tool_prediction,
                        realized_gain=0.0,
                        tool_cost=TOOL_COSTS[tool],
                        source_observation=source_observation,
                        base_execution_threshold=float("inf"),
                        tool_execution_threshold=0.0,
                    )
                    for base, tool_prediction, source_observation in zip(
                        state[split]["base"],
                        state[split][tool],
                        raw["base"][split],
                    )
                ],
                splits[split],
                tool,
            )
            for tool in TOOLS
        }
        source_scores[split] = {
            tool: [row.expected_value for row in source_model.predict(per_tool[tool])]
            for tool in TOOLS
        }
        no_source_scores[split] = {
            tool: [
                row.expected_value
                for row in no_source_model.predict(
                    [_remove_external_context(value) for value in per_tool[tool]]
                )
            ]
            for tool in TOOLS
        }

    labels = {
        split: [target.worthy for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    gains = {
        split: [target.gain for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    methods = {
        "source_multitool_expected": source_scores,
        "no_source_multitool_expected": no_source_scores,
        "fixed_high_fidelity_expected": {
            split: {"high_fidelity": source_scores[split]["high_fidelity"]}
            for split in ("feedback", "acceptance", "test")
        },
        "static_reliability_high_fidelity": {
            split: {
                "high_fidelity": [
                    float(record.features.get("ood_score", 0.0))
                    + float(record.features.get("evidence_conflict", 0.0))
                    + 1.0
                    - float(record.features.get("source_reliability", 0.0))
                    for record in splits[split]
                ]
            }
            for split in ("feedback", "acceptance", "test")
        },
    }
    policies = {}
    for method, scores_by_split in methods.items():
        tools = tuple(scores_by_split["feedback"])
        costs = {tool: TOOL_COSTS[tool] for tool in tools}
        predictions = {
            split: {tool: state[split][tool] for tool in tools}
            for split in ("feedback", "acceptance", "test")
        }
        contract = select_certified_tool_contract(
            scores_by_split["feedback"],
            scores_by_split["acceptance"],
            predictions["feedback"],
            predictions["acceptance"],
            labels["feedback"],
            labels["acceptance"],
            gains["feedback"],
            gains["acceptance"],
            tool_costs=costs,
            acquisition_coverages=COVERAGES,
            execution_coverages=COVERAGES,
            alpha=0.20,
            familywise_delta=0.05,
            baseline_tool=("high_fidelity" if "high_fidelity" in tools else None),
            baseline_weight=0.90,
            max_empirical_risk=0.20,
            selection_mode=policy_selection_mode,
        )
        if contract is None:
            policies[method] = {
                "feedback_policy": None,
                "acceptance_certificate": None,
                "test_evaluation": None,
                "contract_evolution_trace": [],
                "selected_tools": [],
                "deployed": False,
                "deployed_test_population_gain": 0.0,
            }
            continue
        selected_tools = contract.selected_tools
        selected_costs = {tool: costs[tool] for tool in selected_tools}
        test = evaluate_multi_tool_policy(
            contract.policy,
            {tool: scores_by_split["test"][tool] for tool in selected_tools},
            {tool: predictions["test"][tool] for tool in selected_tools},
            labels["test"],
            gains["test"],
            tool_costs=selected_costs,
            alpha=0.20,
            delta=0.05,
        )
        policies[method] = {
            "feedback_policy": contract.policy.to_json(),
            "acceptance_certificate": contract.acceptance.to_json(),
            "test_evaluation": test.to_json(),
            "contract_evolution_trace": list(contract.trace),
            "selected_tools": list(selected_tools),
            "familywise_delta": contract.familywise_delta,
            "deployed": True,
            "deployed_test_population_gain": test.population_gain_after_cost,
        }
    return {
        "experiment": experiment_name,
        "development_only": development_only,
        "task": task,
        "numeric_output_contract": {
            "per_tool_outputs": [f"expected_voi::{tool}" for tool in TOOLS],
            "selected_route_emitted_by_estimator": False,
        },
        "tool_costs": TOOL_COSTS,
        "policy_selection_mode": policy_selection_mode,
        "crossfit_folds": crossfit_folds,
        "cross_fitted_provisional_execution_thresholds": provisional_thresholds,
        "policies": policies,
    }


def _augment_observations(observations, records, tool):
    output = []
    for observation, record in zip(observations, records):
        output.append(
            replace(
                observation,
                context_evidence_support=float(record.features.get("evidence_support", 0.0)),
                context_evidence_conflict=float(record.features.get("evidence_conflict", 0.0)),
                context_ood_score=float(record.features.get("ood_score", 0.0)),
                context_source_reliability=float(record.features.get("source_reliability", 0.0)),
                context_perturbation_stability=float(
                    record.features.get("perturbation_stability", 0.0)
                ),
                candidate_tool_composition=float(tool == "composition"),
                candidate_tool_structure=float(tool == "structure"),
                candidate_tool_high_fidelity=float(tool == "high_fidelity"),
            )
        )
    return output


def _remove_external_context(observation):
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


def _unpool_scores(predictions):
    count = len(predictions) // len(TOOLS)
    return {
        tool: [row.expected_value for row in predictions[index * count : (index + 1) * count]]
        for index, tool in enumerate(TOOLS)
    }


def _markdown(result):
    lines = [
        f"# {result['experiment']}",
        "",
        "The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.",
        "",
        "| method | certified | test gain | selected tools on test | executed tools on test |",
        "|---|---|---:|---|---|",
    ]
    for method, row in result["policies"].items():
        test = row["test_evaluation"] or {
            "selected_tool_counts": {},
            "executed_tool_counts": {},
        }
        lines.append(
            f"| `{method}` | `{row['deployed']}` | "
            f"{row['deployed_test_population_gain']:+.6f} | "
            f"`{json.dumps(test['selected_tool_counts'], sort_keys=True)}` | "
            f"`{json.dumps(test['executed_tool_counts'], sort_keys=True)}` |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
