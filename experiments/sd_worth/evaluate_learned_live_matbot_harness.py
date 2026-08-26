from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression

from harness_matsci.matbot_trajectory import load_matbot_trajectories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--test-trajectories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    trajectories = load_matbot_trajectories(args.trajectories)
    rows = [_learned_row(trajectory) for trajectory in trajectories]
    if args.test_trajectories:
        train, calibration = _two_way_task_split(rows, args.seed)
        evaluation = [
            _learned_row(trajectory)
            for trajectory in load_matbot_trajectories(args.test_trajectories)
        ]
        evaluation_mode = "fresh_external_test"
    else:
        train, calibration, evaluation = _three_way_task_split(rows, args.seed)
        evaluation_mode = "development_holdout"
    task_models = {}
    for task in sorted({row["benchmark"] for row in rows}):
        task_models[task] = _fit_task_model(
            [row for row in train if row["benchmark"] == task],
            [row for row in calibration if row["benchmark"] == task],
            alpha=args.alpha,
            seed=args.seed,
        )
    learned_predictions = [_predict(row, task_models[row["benchmark"]]) for row in evaluation]
    learned_global = _evaluate_predictions(
        learned_predictions,
        budget_fraction=args.budget_fraction,
        acquire_all_evidence=True,
        task_balanced=False,
    )
    learned_balanced = _evaluate_predictions(
        learned_predictions,
        budget_fraction=args.budget_fraction,
        acquire_all_evidence=True,
        task_balanced=True,
    )
    baselines = {}
    for method in ("static_reliability", "base_numeric", "real_tool_numeric"):
        baseline_predictions = [
            {
                **row,
                "lower_gain": float(row["scores"][method]),
                "positive_probability": _rank_probability(row["scores"][method]),
            }
            for row in evaluation
        ]
        baselines[method] = _evaluate_predictions(
            baseline_predictions,
            budget_fraction=args.budget_fraction,
            acquire_all_evidence=method == "real_tool_numeric",
            task_balanced=False,
        )
    report = {
        "schema_version": "learned-live-matbot-harness-v1",
        "input": str(Path(args.trajectories).resolve()),
        "test_input": (
            str(Path(args.test_trajectories).resolve()) if args.test_trajectories else None
        ),
        "evaluation_mode": evaluation_mode,
        "sizes": {"train": len(train), "calibration": len(calibration), "evaluation": len(evaluation)},
        "budget_fraction": args.budget_fraction,
        "alpha": args.alpha,
        "method": {
            "target": "strict realized execution gain after observing actually executed tool evidence",
            "model": "task-specific ExtraTrees gain and success heads",
            "calibration": "task-specific Platt scaling plus one-sided normalized split conformal gain",
            "output": "numeric probability, expected gain, variance, and conformal lower gain; no route",
        },
        "task_models": {task: _model_report(model) for task, model in task_models.items()},
        "learned_global": learned_global,
        "learned_task_balanced_stress_test": learned_balanced,
        "baselines": baselines,
        "claim_boundary": (
            "Fresh fingerprint-disjoint external test; development trajectories are used only "
            "for fitting and calibration."
            if args.test_trajectories
            else "Development split of live local tool executions over historical candidates. "
            "A new fingerprint-disjoint confirmatory batch is still required."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _learned_row(trajectory) -> dict[str, Any]:
    base, tool = trajectory.steps
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
    row = {
        "record_id": trajectory.trajectory_id.split("::", 2)[2],
        "benchmark": benchmark,
        "label": int(realized_gain > 0.0),
        "realized_gain": realized_gain,
        "tool_cost_penalty": 0.15 * float(base.realized_cost or 0.0),
        "scores": {
            "static_reliability": float(sum(static_values) / len(static_values)),
            "base_numeric": float(base.uncertainty_output["expected_strict_net_gain"]),
            "real_tool_numeric": float(tool.uncertainty_output["expected_strict_net_gain"]),
        },
    }
    observation = tool.tool_trace[0]["outputs"]
    feature_map = {
        **{f"internal::{key}": value for key, value in base.internal_signals.items()},
        **{f"external::{key}": value for key, value in base.external_signals.items()},
        **{f"tool::{key}": value for key, value in observation.items()},
        **{f"cost::{key}": value for key, value in base.cost_signals.items()},
    }
    row["feature_map"] = {key: float(value) for key, value in feature_map.items()}
    return row


def _three_way_task_split(rows: list[dict[str, Any]], seed: int):
    train, calibration, evaluation = [], [], []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    for task, task_rows in grouped.items():
        ordered = sorted(
            task_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}|live-tool|{task}|{row['record_id']}".encode()
            ).hexdigest(),
        )
        train_end = int(0.60 * len(ordered))
        calibration_end = int(0.80 * len(ordered))
        train.extend(ordered[:train_end])
        calibration.extend(ordered[train_end:calibration_end])
        evaluation.extend(ordered[calibration_end:])
    return train, calibration, evaluation


def _two_way_task_split(rows: list[dict[str, Any]], seed: int):
    train, calibration = [], []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    for task, task_rows in grouped.items():
        ordered = sorted(
            task_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}|live-tool-confirmatory|{task}|{row['record_id']}".encode()
            ).hexdigest(),
        )
        cut = int(0.75 * len(ordered))
        train.extend(ordered[:cut])
        calibration.extend(ordered[cut:])
    return train, calibration


def _fit_task_model(train, calibration, *, alpha: float, seed: int) -> dict[str, Any]:
    keys = sorted({key for row in train + calibration for key in row["feature_map"]})
    train_x = _matrix(train, keys)
    train_y = np.asarray([row["realized_gain"] for row in train], dtype=float)
    train_labels = np.asarray([row["label"] for row in train], dtype=int)
    classifier = ExtraTreesClassifier(
        n_estimators=384,
        min_samples_leaf=3,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(train_x, train_labels)
    regressor = ExtraTreesRegressor(
        n_estimators=384,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=seed + 1,
        n_jobs=-1,
    ).fit(train_x, train_y)
    calibration_x = _matrix(calibration, keys)
    raw_probabilities = classifier.predict_proba(calibration_x)[:, 1]
    logits = np.asarray([[_logit(value)] for value in raw_probabilities])
    calibration_labels = np.asarray([row["label"] for row in calibration], dtype=int)
    platt = LogisticRegression(C=1.0, solver="lbfgs").fit(logits, calibration_labels)
    tree_predictions = np.asarray(
        [tree.predict(calibration_x) for tree in regressor.estimators_], dtype=float
    )
    means = tree_predictions.mean(axis=0)
    tree_std = tree_predictions.std(axis=0, ddof=1)
    train_tree_predictions = np.asarray(
        [tree.predict(train_x) for tree in regressor.estimators_], dtype=float
    )
    residual_scale = float(
        max(1e-6, np.sqrt(np.mean((train_tree_predictions.mean(axis=0) - train_y) ** 2)))
    )
    predictive_std = np.sqrt(tree_std**2 + residual_scale**2)
    gains = np.asarray([row["realized_gain"] for row in calibration], dtype=float)
    residuals = (means - gains) / np.maximum(1e-6, predictive_std)
    level = min(1.0, math.ceil((len(calibration) + 1) * (1.0 - alpha)) / len(calibration))
    conformal_quantile = float(np.quantile(residuals, level, method="higher"))
    importances = sorted(
        zip(keys, regressor.feature_importances_), key=lambda item: item[1], reverse=True
    )
    return {
        "keys": keys,
        "classifier": classifier,
        "regressor": regressor,
        "platt": platt,
        "residual_scale": residual_scale,
        "conformal_quantile": conformal_quantile,
        "top_feature_importances": [(key, float(value)) for key, value in importances[:15]],
        "train_positive_rate": float(train_labels.mean()),
        "calibration_positive_rate": float(calibration_labels.mean()),
    }


def _predict(row, model) -> dict[str, Any]:
    matrix = _matrix([row], model["keys"])
    raw_probability = float(model["classifier"].predict_proba(matrix)[0, 1])
    calibrated_probability = float(
        model["platt"].predict_proba([[_logit(raw_probability)]])[0, 1]
    )
    tree_predictions = np.asarray(
        [tree.predict(matrix)[0] for tree in model["regressor"].estimators_], dtype=float
    )
    expected_gain = float(tree_predictions.mean())
    predictive_std = float(
        math.sqrt(tree_predictions.std(ddof=1) ** 2 + model["residual_scale"] ** 2)
    )
    return {
        **row,
        "positive_probability": calibrated_probability,
        "expected_gain": expected_gain,
        "predictive_standard_deviation": predictive_std,
        "lower_gain": expected_gain - model["conformal_quantile"] * predictive_std,
    }


def _evaluate_predictions(predictions, *, budget_fraction, acquire_all_evidence, task_balanced):
    if task_balanced:
        selected = []
        for task in sorted({row["benchmark"] for row in predictions}):
            task_rows = [row for row in predictions if row["benchmark"] == task]
            budget = max(1, math.ceil(len(task_rows) * budget_fraction))
            selected.extend(
                row
                for row in sorted(task_rows, key=lambda row: row["lower_gain"], reverse=True)[:budget]
                if row["lower_gain"] > 0.0
            )
        budget = sum(max(1, math.ceil(len([r for r in predictions if r["benchmark"] == task]) * budget_fraction)) for task in {r["benchmark"] for r in predictions})
    else:
        budget = max(1, math.ceil(len(predictions) * budget_fraction))
        selected = [
            row
            for row in sorted(predictions, key=lambda row: row["lower_gain"], reverse=True)[:budget]
            if row["lower_gain"] > 0.0
        ]
    errors = sum(1 - row["label"] for row in selected)
    evidence_cost = sum(row["tool_cost_penalty"] for row in predictions) if acquire_all_evidence else 0.0
    task_breakdown = {}
    for task in sorted({row["benchmark"] for row in predictions}):
        task_rows = [row for row in predictions if row["benchmark"] == task]
        task_selected = [row for row in selected if row["benchmark"] == task]
        task_errors = sum(1 - row["label"] for row in task_selected)
        task_breakdown[task] = {
            "count": len(task_rows),
            "selected": len(task_selected),
            "selective_risk": task_errors / len(task_selected) if task_selected else 0.0,
            "selected_gain": sum(row["realized_gain"] for row in task_selected),
            "positive_count": sum(row["label"] for row in task_rows),
        }
    selected_gain = sum(row["realized_gain"] for row in selected)
    return {
        "budget": budget,
        "selected": len(selected),
        "coverage": len(selected) / len(predictions),
        "errors": errors,
        "selective_risk": errors / len(selected) if selected else 0.0,
        "risk_upper_bound_95": _cp_upper(errors, len(selected)),
        "selected_total_strict_gain": selected_gain,
        "evidence_cost_penalty": evidence_cost,
        "trajectory_total_strict_gain": selected_gain - evidence_cost,
        "task_breakdown": task_breakdown,
    }


def _matrix(rows, keys):
    return np.asarray([[row["feature_map"].get(key, 0.0) for key in keys] for row in rows], dtype=float)


def _model_report(model):
    return {
        "feature_count": len(model["keys"]),
        "train_positive_rate": model["train_positive_rate"],
        "calibration_positive_rate": model["calibration_positive_rate"],
        "residual_scale": model["residual_scale"],
        "conformal_quantile": model["conformal_quantile"],
        "top_feature_importances": model["top_feature_importances"],
    }


def _rank_probability(value):
    return 1.0 / (1.0 + math.exp(-float(value)))


def _logit(value):
    value = min(1.0 - 1e-6, max(1e-6, float(value)))
    return math.log(value / (1.0 - value))


def _cp_upper(errors, total):
    if total == 0:
        return 1.0
    if errors == total:
        return 1.0
    return float(beta.ppf(0.95, errors + 1, total - errors))


if __name__ == "__main__":
    raise SystemExit(main())
