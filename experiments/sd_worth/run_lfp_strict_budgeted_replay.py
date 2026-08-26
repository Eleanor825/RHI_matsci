from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from harness_matsci.budgeted_gain_policy import (
    BudgetedGainPolicy,
    evaluate_budgeted_gain_policy,
)
from harness_matsci.conformal_gain import ConformalGainObservation, fit_split_conformal_gain


BUDGET_FRACTION = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.trajectories).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = run_replay(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))


def run_replay(trajectories):
    if not trajectories:
        raise ValueError("real replay requires trajectories")
    if any(not row.get("real_scientific_outcome") for row in trajectories):
        raise ValueError("all replay rows must have real scientific outcomes")
    folds = sorted({str(row["fold"]) for row in trajectories})
    predictions = []
    fold_reports = []
    for fold in folds:
        calibration = [row for row in trajectories if str(row["fold"]) != fold]
        held = [row for row in trajectories if str(row["fold"]) == fold]
        calibrator = fit_split_conformal_gain(
            [_observation(row) for row in calibration],
            alpha=0.10,
            minimum_scale=1e-3,
        )
        held_rows = []
        for row in held:
            mean, scale = _mean_scale(row)
            lower = calibrator.predict_one(
                record_id=row["record_id"],
                predicted_mean_gain=mean,
                predicted_scale=scale,
            ).lower_gain_bound
            held_rows.append(
                {
                    "record_id": row["record_id"],
                    "fold": fold,
                    "benchmark": row["benchmark"],
                    "actual_gain": _actual_gain(row),
                    "factorized_expected_gain": mean,
                    "factorized_conformal_lower_gain": lower,
                    "direct_llm_judge": _direct_probability(row),
                    "internal_variance": row["steps"][0]["external_tool_evidence"]["internal_variance"],
                    "external_variance": row["steps"][0]["external_tool_evidence"]["external_variance"],
                    "interaction_variance": row["steps"][0]["external_tool_evidence"]["interaction_variance"],
                    "covered": _actual_gain(row) >= lower,
                }
            )
        predictions.extend(held_rows)
        fold_reports.append(_evaluate_rows(held_rows, fold=fold))
    summary = _evaluate_rows(predictions, fold="pooled_cross_fitted")
    summary["conformal_coverage"] = sum(row["covered"] for row in predictions) / len(predictions)
    summary["variance_error_correlations"] = _variance_error_correlations(predictions)
    return {
        "schema_version": "lfp-strict-budgeted-controlled-replay-v1",
        "claim_boundary": (
            "Cross-fitted controlled replay with real historical LFP measurements; "
            "not live MatBot execution and not an independent acceptance certificate."
        ),
        "trajectory_count": len(trajectories),
        "folds": folds,
        "budget_fraction": BUDGET_FRACTION,
        "fold_reports": fold_reports,
        "summary": summary,
        "predictions": predictions,
    }


def _evaluate_rows(rows, *, fold):
    gains = [float(row["actual_gain"]) for row in rows]
    policies = {
        "factorized_conformal_lower_gain": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.0
        ),
        "factorized_expected_gain": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.0
        ),
        "direct_llm_judge": BudgetedGainPolicy(
            BUDGET_FRACTION, minimum_score=0.5
        ),
    }
    return {
        "fold": fold,
        "record_count": len(rows),
        "methods": {
            method: evaluate_budgeted_gain_policy(
                policy,
                [float(row[method]) for row in rows],
                gains,
            )
            for method, policy in policies.items()
        },
    }


def _observation(row):
    mean, scale = _mean_scale(row)
    return ConformalGainObservation(
        record_id=row["record_id"],
        predicted_mean_gain=mean,
        predicted_scale=scale,
        observed_gain=_actual_gain(row),
    )


def _mean_scale(row):
    evidence = row["steps"][0]["external_tool_evidence"]
    variance = (
        float(evidence["internal_variance"])
        + float(evidence["external_variance"])
        + float(evidence["interaction_variance"])
    )
    return float(evidence["fused_mean_gain"]), math.sqrt(max(1e-6, variance))


def _direct_probability(row):
    replicas = row["steps"][0]["internal_model_replicas"]
    return sum(float(cell["p_proposed_action_better"]) for cell in replicas) / len(replicas)


def _actual_gain(row):
    return float(
        row["terminal_outcome_for_training_and_evaluation_only"]["realized_net_gain"]
    )


def _variance_error_correlations(rows):
    squared = [
        (float(row["factorized_expected_gain"]) - float(row["actual_gain"])) ** 2
        for row in rows
    ]
    return {
        name: _correlation([float(row[name]) for row in rows], squared)
        for name in ("internal_variance", "external_variance", "interaction_variance")
    }


def _correlation(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
