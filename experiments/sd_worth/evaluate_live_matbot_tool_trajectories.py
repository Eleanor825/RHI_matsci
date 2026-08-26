from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

from harness_matsci.matbot_trajectory import load_matbot_trajectories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    trajectories = load_matbot_trajectories(args.trajectories)
    rows = [_row(trajectory) for trajectory in trajectories]
    calibration, evaluation = _split(rows, args.seed)
    methods = {}
    for method in ("static_reliability", "base_numeric", "real_tool_numeric"):
        calibrator = _fit_calibrator(calibration, method, args.alpha)
        methods[method] = _evaluate(
            evaluation,
            method,
            calibrator,
            budget_fraction=args.budget_fraction,
            alpha=args.alpha,
        )
        methods[method]["calibrator"] = calibrator
    report = {
        "schema_version": "live-matbot-tool-evaluation-v1",
        "input": str(Path(args.trajectories).resolve()),
        "sizes": {"total": len(rows), "calibration": len(calibration), "evaluation": len(evaluation)},
        "budget_fraction": args.budget_fraction,
        "alpha": args.alpha,
        "methods": methods,
        "claim_boundary": (
            "Development evaluation on a deterministic outcome-blind split of live local tool "
            "executions over historical candidates; not the final confirmatory test."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _row(trajectory) -> dict[str, Any]:
    if len(trajectory.steps) != 2:
        raise ValueError("live tool evaluation requires two-step trajectories")
    base, tool = trajectory.steps
    if not trajectory.is_live_runtime or not trajectory.complete:
        raise ValueError("evaluation requires complete live_runtime trajectories")
    outcome = tool.observed_outcome
    realized_gain = float(outcome["strict_realized_net_gain"])
    benchmark = trajectory.trajectory_id.split("::", 2)[1]
    static_values = (
        base.internal_signals.get("verbal_confidence", 0.5),
        base.internal_signals.get("perturbation_stability", 0.5),
        base.internal_signals.get("estimate_agreement", 0.5),
        base.external_signals.get("evidence_support", 0.5),
        base.external_signals.get("source_reliability", 0.5),
        1.0 - base.external_signals.get("ood_score", 0.5),
        1.0 - base.external_signals.get("evidence_conflict", 0.0),
        1.0 - base.external_signals.get("source_risk", 0.5),
    )
    tool_observation = tool.tool_trace[0]["outputs"]
    return {
        "record_id": trajectory.trajectory_id.split("::", 2)[2],
        "benchmark": benchmark,
        "label": int(realized_gain > 0.0),
        "realized_gain": realized_gain,
        "tool_cost_penalty": 0.15 * float(base.realized_cost or 0.0),
        "parse_success": int(tool_observation.get("parse_success", 1.0) > 0.5),
        "scores": {
            "static_reliability": float(mean(static_values)),
            "base_numeric": float(base.uncertainty_output["expected_strict_net_gain"]),
            "real_tool_numeric": float(tool.uncertainty_output["expected_strict_net_gain"]),
        },
        "standard_deviations": {
            "static_reliability": 1.0,
            "base_numeric": math.sqrt(base.uncertainty_output["total_gain_variance"]),
            "real_tool_numeric": math.sqrt(tool.uncertainty_output["total_gain_variance"]),
        },
    }


def _split(rows: list[dict[str, Any]], seed: int):
    calibration = []
    evaluation = []
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["benchmark"]].append(row)
    for task_rows in by_task.values():
        ordered = sorted(
            task_rows,
            key=lambda row: hashlib.sha256(f"{seed}|{row['record_id']}".encode()).hexdigest(),
        )
        cut = int(round(2.0 * len(ordered) / 3.0))
        calibration.extend(ordered[:cut])
        evaluation.extend(ordered[cut:])
    return calibration, evaluation


def _fit_calibrator(rows: list[dict[str, Any]], method: str, alpha: float) -> dict[str, Any]:
    scores = np.asarray([[row["scores"][method]] for row in rows], dtype=float)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    logistic = LogisticRegression(C=1.0, solver="lbfgs").fit(scores, labels)
    predicted = logistic.predict_proba(scores)[:, 1]
    gains = np.asarray([row["realized_gain"] for row in rows], dtype=float)
    standard_deviations = np.asarray(
        [max(1e-6, row["standard_deviations"][method]) for row in rows], dtype=float
    )
    residuals = (np.asarray([row["scores"][method] for row in rows]) - gains) / standard_deviations
    quantile_level = min(1.0, math.ceil((len(rows) + 1) * (1.0 - alpha)) / len(rows))
    conformal_quantile = float(np.quantile(residuals, quantile_level, method="higher"))
    return {
        "logistic_intercept": float(logistic.intercept_[0]),
        "logistic_coefficient": float(logistic.coef_[0, 0]),
        "conformal_quantile": conformal_quantile,
        "calibration_brier": float(np.mean((predicted - labels) ** 2)),
        "calibration_ece": _ece(predicted, labels),
    }


def _evaluate(
    rows: list[dict[str, Any]],
    method: str,
    calibrator: dict[str, Any],
    *,
    budget_fraction: float,
    alpha: float,
) -> dict[str, Any]:
    coefficient = calibrator["logistic_coefficient"]
    intercept = calibrator["logistic_intercept"]
    conformal = calibrator["conformal_quantile"]
    scored = []
    probabilities = []
    labels = []
    for row in rows:
        raw = row["scores"][method]
        probability = 1.0 / (1.0 + math.exp(-(intercept + coefficient * raw)))
        lower_gain = raw - conformal * row["standard_deviations"][method]
        scored.append((lower_gain, probability, row))
        probabilities.append(probability)
        labels.append(row["label"])
    budget = max(1, math.ceil(len(rows) * budget_fraction))
    selected = [item for item in sorted(scored, key=lambda item: item[0], reverse=True)[:budget] if item[0] > 0]
    errors = sum(1 - item[2]["label"] for item in selected)
    selected_gain = sum(item[2]["realized_gain"] for item in selected)
    evidence_cost = sum(row["tool_cost_penalty"] for row in rows) if method == "real_tool_numeric" else 0.0
    task_breakdown = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        task_rows = [row for row in rows if row["benchmark"] == benchmark]
        task_selected = [item for item in selected if item[2]["benchmark"] == benchmark]
        task_errors = sum(1 - item[2]["label"] for item in task_selected)
        task_breakdown[benchmark] = {
            "count": len(task_rows),
            "selected": len(task_selected),
            "selective_risk": task_errors / len(task_selected) if task_selected else 0.0,
            "selected_gain": sum(item[2]["realized_gain"] for item in task_selected),
            "parse_failure_count": sum(1 - row["parse_success"] for row in task_rows),
        }
    return {
        "selected": len(selected),
        "budget": budget,
        "coverage": len(selected) / len(rows),
        "errors": errors,
        "selective_risk": errors / len(selected) if selected else 0.0,
        "risk_upper_bound_95": _clopper_pearson_upper(errors, len(selected)),
        "selected_total_strict_gain": selected_gain,
        "evidence_cost_penalty": evidence_cost,
        "trajectory_total_strict_gain": selected_gain - evidence_cost,
        "evaluation_brier": float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2)),
        "evaluation_ece": _ece(np.asarray(probabilities), np.asarray(labels)),
        "task_breakdown": task_breakdown,
    }


def _clopper_pearson_upper(errors: int, total: int) -> float:
    if total == 0:
        return 1.0
    if errors == total:
        return 1.0
    return float(beta.ppf(0.95, errors + 1, total - errors))


def _ece(probabilities, labels, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    total = len(labels)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean() - labels[mask].mean()))
    return error


if __name__ == "__main__":
    raise SystemExit(main())
