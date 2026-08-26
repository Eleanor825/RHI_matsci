from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from harness_matsci.external_h14_tools import (
    EVIDENCE_SOURCES,
    ExternalH14EvidenceEnvironment,
)
from harness_matsci.frozen_policy_certificate import (
    certify_frozen_policy,
    select_feedback_policy,
)
from harness_matsci.hurdle_gain_calibration import (
    HurdleGainObservation,
    fit_hurdle_gain_calibrator,
)
from harness_matsci.risk_control import audit_risk_controlled_grid, select_risk_controlled_grid
from harness_matsci.schema import ActionRecord
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    gain_targets,
)
from harness_matsci.source_fusion import (
    SourceFusionObservation,
    fit_source_reliability_fuser,
)
from harness_matsci.source_margin_calibration import fit_source_margin_calibrator


TASKS = (
    "external_pairwise_dielectric",
    "external_unique_jdft2d",
    "external_extreme_phonons",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--raw-margins", action="store_true")
    args = parser.parse_args()
    result = run_source_pilot(
        args.fold_dir,
        args.snapshot_root,
        seed=args.seed,
        calibrate_margins=not args.raw_margins,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


def run_source_pilot(
    fold_dir: str | Path,
    snapshot_root: str | Path,
    *,
    seed: int,
    calibrate_margins: bool = True,
) -> dict:
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
    environment = ExternalH14EvidenceEnvironment.fit(
        snapshot_root,
        splits["train"],
        seed=seed,
    )
    source_observations = {
        split: _source_observations(
            environment,
            records,
            target_by_id,
        )
        for split, records in splits.items()
        if split != "train"
    }
    margin_calibrator = None
    if calibrate_margins:
        margin_calibrator = fit_source_margin_calibrator(
            source_observations["feedback"]
        )
        source_observations = {
            split: margin_calibrator.predict(observations)
            for split, observations in source_observations.items()
        }
    source_fuser = fit_source_reliability_fuser(
        source_observations["feedback"],
        fit_affine=not calibrate_margins,
    )
    summaries = {
        split: [prediction.to_json() for prediction in source_fuser.predict(observations)]
        for split, observations in source_observations.items()
    }
    feedback_observations = _hurdle_observations(
        summaries["feedback"], target_by_id
    )
    source_model = fit_hurdle_gain_calibrator(
        feedback_observations,
        use_source_variance=True,
        beta=0.10,
        tail_weight=0.50,
    )
    no_source_model = fit_hurdle_gain_calibrator(
        feedback_observations,
        use_source_variance=False,
        beta=0.10,
        tail_weight=0.50,
    )
    methods = {}
    methods["direct_fused_mean"] = {
        split: [float(row["mean_gain"]) for row in summaries[split]]
        for split in ("feedback", "acceptance", "test")
    }
    for source in EVIDENCE_SOURCES:
        methods[f"direct_{source}"] = {
            split: [float(row["source_means"][source]) for row in summaries[split]]
            for split in ("feedback", "acceptance", "test")
        }
    for model_name, fitted in (
        ("source", source_model),
        ("no_source", no_source_model),
    ):
        for tail_weight in (0.0, 0.25, 0.50, 1.0):
            weighted = replace(fitted, tail_weight=tail_weight)
            predictions = {
                split: weighted.predict(
                    _hurdle_observations(summaries[split], target_by_id)
                )
                for split in ("feedback", "acceptance", "test")
            }
            for score_name, extractor in (
                ("p", lambda prediction: prediction.p_positive_gain),
                ("mean", lambda prediction: prediction.mean_gain),
                ("tail", lambda prediction: prediction.tail_risk_score),
            ):
                methods[f"{model_name}_{score_name}_{tail_weight:.2f}"] = {
                    split: [extractor(prediction) for prediction in split_predictions]
                    for split, split_predictions in predictions.items()
                }
    labels = {
        split: [target.worthy for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    gains = {
        split: [target.gain for target in targets[split]]
        for split in ("feedback", "acceptance", "test")
    }
    results = {
        method: _evaluate_task_conditional(
            scores,
            splits,
            labels,
            gains,
            tool_cost_per_selected_action=0.024,
        )
        for method, scores in methods.items()
    }
    method_costs = {
        method: (
            0.002
            if method in {"direct_composition", "direct_structure"}
            else 0.020
            if method == "direct_high_fidelity"
            else 0.024
        )
        for method in methods
    }
    frozen_policies = _freeze_and_certify(
        methods,
        splits,
        labels,
        gains,
        method_costs,
    )
    return {
        "experiment": "h14_external_source_decomposition_pilot",
        "development_only": True,
        "acquisition_assumption": "oracle_selected_only_upper_bound",
        "evidence_sources": list(EVIDENCE_SOURCES),
        "tool_cost_per_selected_action": 0.024,
        "margin_calibration": (
            margin_calibrator.to_json() if margin_calibrator is not None else None
        ),
        "source_fuser": source_fuser.to_json(),
        "source_model": source_model.to_json(),
        "no_source_model": no_source_model.to_json(),
        "methods": results,
        "frozen_policies": frozen_policies,
    }


def _source_observations(environment, records, target_by_id):
    views_by_source = {
        source: [environment.observe(record, source=source) for record in records]
        for source in EVIDENCE_SOURCES
    }
    observations = []
    for record_index, record in enumerate(records):
        matrix = [
            [
                float(views_by_source[source][record_index].features["tool_estimated_margin"])
                + replica_offset
                * float(views_by_source[source][record_index].features["tool_uncertainty"])
                for source in EVIDENCE_SOURCES
            ]
            for replica_offset in (-1.0, 0.0, 1.0)
        ]
        observations.append(
            SourceFusionObservation(
                record_id=record.record_id,
                benchmark=record.benchmark,
                source_names=tuple(EVIDENCE_SOURCES),
                predicted_gain=tuple(tuple(row) for row in matrix),
                gain=target_by_id[record.record_id].gain,
            )
        )
    return observations


def _hurdle_observations(rows, target_by_id):
    return [
        HurdleGainObservation(
            record_id=row["record_id"],
            benchmark=row["benchmark"],
            mean_gain=row["mean_gain"],
            internal_variance=row["internal_variance"],
            external_variance=row["external_variance"],
            gain=target_by_id[row["record_id"]].gain,
        )
        for row in rows
    ]


def _evaluate_task_conditional(
    scores,
    splits,
    labels,
    gains,
    *,
    tool_cost_per_selected_action,
):
    thresholds = {}
    acceptance_by_task = {}
    for task in TASKS:
        acceptance_indices = [
            index
            for index, record in enumerate(splits["acceptance"])
            if record.benchmark == task
        ]
        selection = select_risk_controlled_grid(
            [scores["acceptance"][index] for index in acceptance_indices],
            [labels["acceptance"][index] for index in acceptance_indices],
            coverages=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
            alpha=0.20,
            delta=0.05 / len(TASKS),
        )
        thresholds[task] = selection.threshold
        acceptance_by_task[task] = {
            "selection": selection.to_json(),
            "grid": audit_risk_controlled_grid(
                [scores["acceptance"][index] for index in acceptance_indices],
                [labels["acceptance"][index] for index in acceptance_indices],
                coverages=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=0.20,
                delta=0.05 / len(TASKS),
            ),
        }
    test = _evaluate_thresholds(
        scores["test"],
        splits["test"],
        labels["test"],
        gains["test"],
        thresholds,
        tool_cost_per_selected_action,
    )
    return {
        "thresholds": thresholds,
        "acceptance_by_task": acceptance_by_task,
        "test": test,
    }


def _evaluate_thresholds(scores, records, labels, gains, thresholds, tool_cost):
    output = {}
    total_gain = 0.0
    total_selected = 0
    for task in TASKS:
        indices = [index for index, record in enumerate(records) if record.benchmark == task]
        selected = [index for index in indices if scores[index] >= thresholds[task]]
        selected_gain = sum(gains[index] for index in selected)
        total_gain += selected_gain - tool_cost * len(selected)
        total_selected += len(selected)
        output[task] = {
            "selected_count": len(selected),
            "coverage": len(selected) / len(indices),
            "selective_risk": (
                sum(1 - labels[index] for index in selected) / len(selected)
                if selected
                else 0.0
            ),
            "selected_mean_gain": selected_gain / len(selected) if selected else 0.0,
            "population_gain_after_cost": (
                selected_gain - tool_cost * len(selected)
            )
            / len(indices),
        }
    output["aggregate"] = {
        "selected_count": total_selected,
        "coverage": total_selected / len(records),
        "population_gain_after_cost": total_gain / len(records),
    }
    return output


def _freeze_and_certify(methods, splits, labels, gains, method_costs):
    families = {
        "proposed_source_tail": tuple(
            f"source_tail_{tail_weight:.2f}"
            for tail_weight in (0.0, 0.25, 0.50, 1.0)
        ),
        "matched_no_source_tail": tuple(
            f"no_source_tail_{tail_weight:.2f}"
            for tail_weight in (0.0, 0.25, 0.50, 1.0)
        ),
        "direct_high_fidelity": ("direct_high_fidelity",),
        "direct_fused_mean": ("direct_fused_mean",),
    }
    output = {}
    for family, family_methods in families.items():
        output[family] = {}
        for task in TASKS:
            indices = {
                split: [
                    index
                    for index, record in enumerate(splits[split])
                    if record.benchmark == task
                ]
                for split in ("feedback", "acceptance", "test")
            }
            feedback_scores = {
                method: [methods[method]["feedback"][index] for index in indices["feedback"]]
                for method in family_methods
            }
            task_costs = {method: method_costs[method] for method in family_methods}
            policy = select_feedback_policy(
                feedback_scores,
                [labels["feedback"][index] for index in indices["feedback"]],
                [gains["feedback"][index] for index in indices["feedback"]],
                task_costs,
                coverages=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                max_empirical_risk=0.20,
            )
            if policy.method == "no_op":
                acceptance_scores = [0.0] * len(indices["acceptance"])
                test_scores = [0.0] * len(indices["test"])
                tool_cost = 0.0
            else:
                acceptance_scores = [
                    methods[policy.method]["acceptance"][index]
                    for index in indices["acceptance"]
                ]
                test_scores = [
                    methods[policy.method]["test"][index]
                    for index in indices["test"]
                ]
                tool_cost = method_costs[policy.method]
            acceptance = certify_frozen_policy(
                policy,
                acceptance_scores,
                [labels["acceptance"][index] for index in indices["acceptance"]],
                [gains["acceptance"][index] for index in indices["acceptance"]],
                tool_cost=tool_cost,
                alpha=0.20,
                delta=0.05 / len(TASKS),
            )
            test = certify_frozen_policy(
                policy,
                test_scores,
                [labels["test"][index] for index in indices["test"]],
                [gains["test"][index] for index in indices["test"]],
                tool_cost=tool_cost,
                alpha=0.20,
                delta=0.05 / len(TASKS),
            )
            output[family][task] = {
                "feedback_policy": policy.to_json(),
                "acceptance_certificate": acceptance.to_json(),
                "test_evaluation": test.to_json(),
                "deployed": acceptance.certified,
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
        "# H14 External Source-Decomposition Pilot",
        "",
        "Fold 0 development only. Costs assume evidence is acquired only for actions",
        "eventually selected, so this is an upper bound for the learned VoE policy.",
        "",
        "## Feedback-Frozen Policies",
        "",
        "Feedback selects one method and threshold; acceptance certifies it once;",
        "test copies it unchanged.",
        "",
        "| family | task | method | feedback coverage | acceptance n | acceptance risk | risk UCB | deployed | test coverage | test risk | test gain |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for family, tasks in result["frozen_policies"].items():
        for task, values in tasks.items():
            policy = values["feedback_policy"]
            acceptance = values["acceptance_certificate"]
            test = values["test_evaluation"]
            lines.append(
                f"| `{family}` | `{task}` | `{policy['method']}` | "
                f"{policy['target_coverage']:.4f} | {acceptance['selected_count']} | "
                f"{acceptance['empirical_risk']:.4f} | "
                f"{acceptance['risk_upper_bound']:.4f} | "
                f"`{values['deployed']}` | {test['coverage']:.4f} | "
                f"{test['empirical_risk']:.4f} | "
                f"{test['population_gain_after_cost']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Same-Split Grid Diagnostic",
            "",
            "This legacy diagnostic searches acceptance prefixes and is deliberately",
            "more conservative than the feedback-frozen protocol above.",
            "",
        "| method | aggregate coverage | aggregate gain | pair gain | unique gain | extreme gain |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        result["methods"].items(),
        key=lambda item: item[1]["test"]["aggregate"]["population_gain_after_cost"],
        reverse=True,
    )
    for method, values in ranked:
        test = values["test"]
        lines.append(
            f"| `{method}` | {test['aggregate']['coverage']:.4f} | "
            f"{test['aggregate']['population_gain_after_cost']:+.6f} | "
            f"{test['external_pairwise_dielectric']['population_gain_after_cost']:+.6f} | "
            f"{test['external_unique_jdft2d']['population_gain_after_cost']:+.6f} | "
            f"{test['external_extreme_phonons']['population_gain_after_cost']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
