from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from harness_matsci.evidence_value import (
    EvidenceValueObservation,
    cross_fitted_evidence_value_predictions,
    evidence_value_observation,
    fit_evidence_value_model,
)
from harness_matsci.external_h14_tools import ExternalH14EvidenceEnvironment
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
    select_two_stage_policy,
)


TASKS = (
    "external_pairwise_dielectric",
    "external_unique_jdft2d",
    "external_extreme_phonons",
)
BASE_SOURCES = ("composition", "structure")
TOOL_SOURCES = ("composition", "structure", "high_fidelity")
TOOL_COST = 0.020
ACQUISITION_COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    args = parser.parse_args()
    result = run_voe_pilot(
        args.fold_dir,
        args.snapshot_root,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


def run_voe_pilot(
    fold_dir: str | Path,
    snapshot_root: str | Path,
    *,
    seed: int,
    crossfit_folds: int,
) -> dict[str, object]:
    root = Path(fold_dir)
    splits = {
        split: [
            record
            for task in TASKS
            for record in _load_records(root / f"{task}.{split}.jsonl")
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
    gain_projector = EstimatedGainProjector(normalizer, reservations, config)
    environment = ExternalH14EvidenceEnvironment.fit(
        snapshot_root,
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
                gain_projector=gain_projector,
            )
            for split, records in splits.items()
            if split != "train"
        }
        for state, sources in (("base", BASE_SOURCES), ("tool", TOOL_SOURCES))
    }

    feedback_base = cross_fitted_source_state_predictions(
        raw["base"]["feedback"],
        folds=crossfit_folds,
        seed=seed + 11,
        use_source_variance=True,
        tail_weight=0.50,
        calibrate_source_margins=False,
    )
    feedback_tool = cross_fitted_source_state_predictions(
        raw["tool"]["feedback"],
        folds=crossfit_folds,
        seed=seed + 29,
        use_source_variance=True,
        tail_weight=0.50,
        calibrate_source_margins=False,
    )
    feedback_voe = [
        evidence_value_observation(
            base,
            tool,
            realized_gain=target_by_id[base.record_id].gain,
            tool_cost=TOOL_COST,
        )
        for base, tool in zip(feedback_base, feedback_tool)
    ]
    feedback_voe_no_source = [
        replace(observation, base_external_variance=0.0)
        for observation in feedback_voe
    ]
    feedback_acquisition = {
        "source_voi": cross_fitted_evidence_value_predictions(
            feedback_voe,
            seed=seed + 101,
            folds=crossfit_folds,
            replicas=3,
            trees=80,
        ),
        "no_source_voi": cross_fitted_evidence_value_predictions(
            feedback_voe_no_source,
            seed=seed + 211,
            folds=crossfit_folds,
            replicas=3,
            trees=80,
        ),
    }

    base_model = fit_source_state_model(
        raw["base"]["feedback"],
        use_source_variance=True,
        tail_weight=0.50,
        calibrate_source_margins=False,
    )
    tool_model = fit_source_state_model(
        raw["tool"]["feedback"],
        use_source_variance=True,
        tail_weight=0.50,
        calibrate_source_margins=False,
    )
    source_voe_model = fit_evidence_value_model(
        feedback_voe,
        seed=seed + 307,
        replicas=5,
        trees=160,
    )
    no_source_voe_model = fit_evidence_value_model(
        feedback_voe_no_source,
        seed=seed + 401,
        replicas=5,
        trees=160,
    )

    state_predictions = {
        "feedback": {"base": feedback_base, "tool": feedback_tool}
    }
    acquisition_scores = {
        "feedback": {
            "source_voi_expected": [
                row.expected_value for row in feedback_acquisition["source_voi"]
            ],
            "source_voi_probability": [
                row.positive_value_probability
                for row in feedback_acquisition["source_voi"]
            ],
            "no_source_voi_expected": [
                row.expected_value for row in feedback_acquisition["no_source_voi"]
            ],
            "no_source_voi_probability": [
                row.positive_value_probability
                for row in feedback_acquisition["no_source_voi"]
            ],
            "external_variance": [
                row.calibrated_external_variance for row in feedback_base
            ],
            "predictive_variance": [row.calibrated_variance for row in feedback_base],
        }
    }
    for split in ("acceptance", "test"):
        base_predictions = base_model.predict(raw["base"][split])
        tool_predictions = tool_model.predict(raw["tool"][split])
        state_predictions[split] = {
            "base": base_predictions,
            "tool": tool_predictions,
        }
        source_observations = [
            _prediction_observation(base, tool_cost=TOOL_COST)
            for base in base_predictions
        ]
        no_source_observations = [
            replace(observation, base_external_variance=0.0)
            for observation in source_observations
        ]
        acquisition_scores[split] = {
            "source_voi_expected": [],
            "source_voi_probability": [],
            "no_source_voi_expected": [],
            "no_source_voi_probability": [],
            "external_variance": [
                row.calibrated_external_variance for row in base_predictions
            ],
            "predictive_variance": [row.calibrated_variance for row in base_predictions],
        }
        source_predictions = source_voe_model.predict(source_observations)
        no_source_predictions = no_source_voe_model.predict(no_source_observations)
        acquisition_scores[split]["source_voi_expected"] = [
            row.expected_value for row in source_predictions
        ]
        acquisition_scores[split]["source_voi_probability"] = [
            row.positive_value_probability for row in source_predictions
        ]
        acquisition_scores[split]["no_source_voi_expected"] = [
            row.expected_value for row in no_source_predictions
        ]
        acquisition_scores[split]["no_source_voi_probability"] = [
            row.positive_value_probability for row in no_source_predictions
        ]

    labels = {
        split: [target.worthy for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    gains = {
        split: [target.gain for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    policies = _fit_and_evaluate_policies(
        splits,
        acquisition_scores,
        state_predictions,
        labels,
        gains,
    )
    return {
        "experiment": "h14_two_stage_tail_risk_value_of_evidence_pilot",
        "development_only": True,
        "fold": 0,
        "crossfit_folds": crossfit_folds,
        "base_sources": list(BASE_SOURCES),
        "tool_sources": list(TOOL_SOURCES),
        "tool_cost": TOOL_COST,
        "source_input_space": "train_only_projected_scientific_gain",
        "gain_projector": gain_projector.to_json(),
        "decision_rule": "acquire by frozen VoE threshold; execute with tool posterior if acquired, otherwise base posterior; tail-risk score > 0",
        "base_model": base_model.to_json(),
        "tool_model": tool_model.to_json(),
        "source_voe_model": source_voe_model.to_json(),
        "no_source_voe_model": no_source_voe_model.to_json(),
        "feedback_realized_voe": _voe_summary(feedback_voe),
        "policies": policies,
    }


def _fit_and_evaluate_policies(
    splits,
    acquisition_scores,
    state_predictions,
    labels,
    gains,
):
    output = {}
    for method in (
        "source_voi_expected",
        "source_voi_probability",
        "no_source_voi_expected",
        "no_source_voi_probability",
        "external_variance",
        "predictive_variance",
    ):
        output[method] = {}
        for task in TASKS:
            indices = {
                split: [
                    index
                    for index, record in enumerate(splits[split])
                    if record.benchmark == task
                ]
                for split in ("feedback", "acceptance", "test")
            }
            policy = select_two_stage_policy(
                [acquisition_scores["feedback"][method][index] for index in indices["feedback"]],
                [state_predictions["feedback"]["base"][index] for index in indices["feedback"]],
                [state_predictions["feedback"]["tool"][index] for index in indices["feedback"]],
                [labels["feedback"][index] for index in indices["feedback"]],
                [gains["feedback"][index] for index in indices["feedback"]],
                tool_cost=TOOL_COST,
                coverages=ACQUISITION_COVERAGES,
                max_empirical_risk=0.20,
            )
            evaluations = {}
            for split in ("acceptance", "test"):
                evaluations[split] = evaluate_two_stage_policy(
                    policy,
                    [acquisition_scores[split][method][index] for index in indices[split]],
                    [state_predictions[split]["base"][index] for index in indices[split]],
                    [state_predictions[split]["tool"][index] for index in indices[split]],
                    [labels[split][index] for index in indices[split]],
                    [gains[split][index] for index in indices[split]],
                    tool_cost=TOOL_COST,
                    alpha=0.20,
                    delta=0.05 / len(TASKS),
                )
            output[method][task] = {
                "feedback_policy": policy.to_json(),
                "acceptance_certificate": evaluations["acceptance"].to_json(),
                "test_evaluation": evaluations["test"].to_json(),
                "deployed": evaluations["acceptance"].certified,
                "deployed_test_population_gain": (
                    evaluations["test"].population_gain_after_cost
                    if evaluations["acceptance"].certified
                    else 0.0
                ),
            }
    return output


def _prediction_observation(prediction, *, tool_cost):
    return EvidenceValueObservation(
        record_id=prediction.record_id,
        benchmark=prediction.benchmark,
        base_mean_gain=prediction.mean_gain,
        base_p_positive_gain=prediction.p_positive_gain,
        base_expected_shortfall=prediction.expected_shortfall,
        base_tail_risk_score=prediction.tail_risk_score,
        base_internal_variance=prediction.calibrated_internal_variance,
        base_external_variance=prediction.calibrated_external_variance,
        tool_cost=tool_cost,
        realized_value=0.0,
    )


def _voe_summary(observations):
    output = {}
    for task in TASKS:
        values = [row.realized_value for row in observations if row.benchmark == task]
        output[task] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "positive_fraction": sum(value > 0.0 for value in values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return output


def _load_records(path: Path):
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _markdown(result):
    lines = [
        "# H14 Two-Stage Tail-Risk Value-of-Evidence Pilot",
        "",
        "Fold 0 is development-only. Feedback state and VoE predictions are",
        "cross-fitted; acceptance certifies one frozen acquisition threshold; test",
        "copies the policy unchanged.",
        "",
        "| method | task | acquire target | acceptance acquired | acceptance executed | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for method, tasks in result["policies"].items():
        for task, values in tasks.items():
            policy = values["feedback_policy"]
            acceptance = values["acceptance_certificate"]
            test = values["test_evaluation"]
            lines.append(
                f"| `{method}` | `{task}` | "
                f"{policy['target_acquisition_coverage']:.4f} | "
                f"{acceptance['acquired_count']} | {acceptance['executed_count']} | "
                f"{acceptance['risk_upper_bound']:.4f} | `{values['deployed']}` | "
                f"{test['acquired_count']} | {test['executed_count']} | "
                f"{test['selective_risk']:.4f} | "
                f"{values['deployed_test_population_gain']:+.6f} |"
            )
    lines.extend(["", "## Feedback Counterfactual Tool Value", ""])
    for task, summary in result["feedback_realized_voe"].items():
        lines.append(
            f"- `{task}`: mean `{summary['mean']:+.6f}`, positive fraction "
            f"`{summary['positive_fraction']:.4f}`, range "
            f"`[{summary['minimum']:+.6f}, {summary['maximum']:+.6f}]`."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
