from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h17_pairwise import TASK
from harness_matsci.external_h17_pairwise_tools import ExternalH17PairwiseEvidenceEnvironment
from harness_matsci.metrics import binary_metrics, fixed_coverage_metrics
from harness_matsci.risk_control import (
    binomial_upper_confidence,
    select_risk_controlled_grid,
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

from run_external_h15_unique_voe import _load_records


SOURCES = ("agent_prior", "composition", "structure")
COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    args = parser.parse_args()
    result = run_ablation(
        args.fold_dir,
        args.source,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(_compact(result), sort_keys=True))


def run_ablation(fold_dir, source, *, seed, crossfit_folds):
    root = Path(fold_dir)
    splits = {
        split: _load_records(root / f"{TASK}.{split}.jsonl")
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
        target.record_id: target for rows in targets.values() for target in rows
    }
    environment = ExternalH17PairwiseEvidenceEnvironment.fit(
        source,
        splits["train"],
        seed=seed,
    )
    observations = {
        split: build_source_observations(
            environment,
            splits[split],
            target_by_id,
            source_names=SOURCES,
            gain_projector=projector,
        )
        for split in ("feedback", "acceptance", "test")
    }
    output = {
        "task": TASK,
        "development_only": True,
        "fold_dir": str(fold_dir),
        "methods": {},
    }
    for name, use_d3 in (("variance_only", False), ("d3_boundary", True)):
        feedback = cross_fitted_source_state_predictions(
            observations["feedback"],
            folds=crossfit_folds,
            seed=seed + 31,
            calibrate_source_margins=False,
            use_decision_instability=use_d3,
        )
        model = fit_source_state_model(
            observations["feedback"],
            calibrate_source_margins=False,
            use_decision_instability=use_d3,
        )
        predictions = {
            "feedback": feedback,
            "acceptance": model.predict(observations["acceptance"]),
            "test": model.predict(observations["test"]),
        }
        labels = {
            split: [target.worthy for target in targets[split]]
            for split in ("feedback", "acceptance", "test")
        }
        gains = {
            split: [target.gain for target in targets[split]]
            for split in ("feedback", "acceptance", "test")
        }
        selection = select_risk_controlled_grid(
            [row.p_positive_gain for row in predictions["feedback"]],
            labels["feedback"],
            coverages=COVERAGES,
            alpha=0.20,
            delta=0.05,
        )
        acceptance = _evaluate_frozen(
            predictions["acceptance"],
            labels["acceptance"],
            gains["acceptance"],
            threshold=selection.threshold,
            alpha=0.20,
            delta=0.05,
        )
        test = _evaluate_frozen(
            predictions["test"],
            labels["test"],
            gains["test"],
            threshold=selection.threshold,
            alpha=0.20,
            delta=0.05,
        )
        output["methods"][name] = {
            "feedback_selection": selection.to_json(),
            "acceptance": acceptance,
            "test": test,
            "deployed_test_population_gain": (
                test["population_gain"] if acceptance["certified"] else 0.0
            ),
            "calibrator": model.hurdle_calibrator.to_json(),
            "raw_state_summary": {
                split: _raw_state_summary(rows) for split, rows in observations.items()
            },
        }
    return output


def _evaluate_frozen(predictions, labels, gains, *, threshold, alpha, delta):
    probabilities = [row.p_positive_gain for row in predictions]
    selected = [index for index, value in enumerate(probabilities) if value >= threshold]
    errors = sum(1 - labels[index] for index in selected)
    risk = errors / len(selected) if selected else 0.0
    upper = binomial_upper_confidence(errors, len(selected), delta) if selected else 0.0
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(labels),
        "selective_risk": risk,
        "risk_upper_bound": upper,
        "certified": bool(selected and upper <= alpha),
        "population_gain": sum(gains[index] for index in selected) / len(labels),
        "mean_selected_gain": (
            sum(gains[index] for index in selected) / len(selected) if selected else 0.0
        ),
        "calibration": binary_metrics(labels, probabilities, threshold),
        "fixed_coverage": fixed_coverage_metrics(labels, probabilities),
    }


def _raw_state_summary(observations):
    from harness_matsci.source_fusion import fit_source_reliability_fuser

    predictions = fit_source_reliability_fuser(
        observations,
        fit_affine=False,
    ).predict(observations)
    return {
        "mean_internal_boundary_instability": _mean(
            row.internal_boundary_instability for row in predictions
        ),
        "mean_external_boundary_instability": _mean(
            row.external_boundary_instability for row in predictions
        ),
        "internal_boundary_nonzero_fraction": _mean(
            float(row.internal_boundary_instability > 0.0) for row in predictions
        ),
        "external_boundary_nonzero_fraction": _mean(
            float(row.external_boundary_instability > 0.0) for row in predictions
        ),
    }


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _compact(result):
    return {
        name: {
            "acceptance_certified": values["acceptance"]["certified"],
            "acceptance_risk_ucb": values["acceptance"]["risk_upper_bound"],
            "test_brier": values["test"]["calibration"]["brier"],
            "test_ece": values["test"]["calibration"]["ece"],
            "test_aurc": values["test"]["calibration"]["aurc"],
            "test_population_gain": values["test"]["population_gain"],
            "deployed_test_population_gain": values["deployed_test_population_gain"],
        }
        for name, values in result["methods"].items()
    }


if __name__ == "__main__":
    main()
