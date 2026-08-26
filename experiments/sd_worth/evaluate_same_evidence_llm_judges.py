from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from harness_matsci.conformal_gain import (
    ConformalGainObservation,
    fit_split_conformal_gain,
)
from harness_matsci.risk_control import binomial_upper_confidence


COST_WEIGHT = 0.15


def evaluate_same_evidence_judges(
    feedback_by_stage: dict[str, dict[str, object]],
    acceptance_by_stage: dict[str, dict[str, object]],
    *,
    budget_fraction: float,
    alpha: float,
) -> dict[str, object]:
    results = {}
    for stage in sorted(feedback_by_stage):
        feedback_rows = feedback_by_stage[stage]["rows"]
        acceptance_rows = acceptance_by_stage[stage]["rows"]
        models = tuple(feedback_by_stage[stage]["models"])
        methods = tuple(models) + ("ensemble",)
        for method in methods:
            platt = _fit_platt(feedback_rows, method)
            conformal = _fit_gain_conformal(feedback_rows, method)
            probabilities = [
                platt[(row["benchmark"], method)].predict(
                    _row_probability(row, method)
                )
                for row in acceptance_rows
            ]
            lower_gains = []
            for row in acceptance_rows:
                mean_gain = _row_expected_gain(row, method)
                bound = conformal[(row["benchmark"], method)].predict_one(
                    record_id=row["record_id"],
                    predicted_mean_gain=mean_gain,
                    predicted_scale=1.0,
                )
                lower_gains.append(bound.lower_gain_bound)
            results[f"{stage}::{method}::platt_p"] = _evaluate_scores(
                acceptance_rows,
                probabilities,
                budget_fraction=budget_fraction,
                positive_threshold=0.5,
                alpha=alpha,
            )
            results[f"{stage}::{method}::conformal_gain"] = _evaluate_scores(
                acceptance_rows,
                lower_gains,
                budget_fraction=budget_fraction,
                positive_threshold=0.0,
                alpha=alpha,
            )
    strongest = max(
        results,
        key=lambda name: (
            results[name]["trajectory_total_strict_gain"],
            -results[name]["selective_risk"],
            name,
        ),
    )
    return {
        "schema_version": "same-evidence-llm-judge-evaluation-v1",
        "budget_fraction": budget_fraction,
        "alpha": alpha,
        "results": results,
        "strongest_method": strongest,
        "strongest_result": results[strongest],
    }


class _Platt:
    def __init__(self, coefficient: float, intercept: float):
        self.coefficient = coefficient
        self.intercept = intercept

    def predict(self, probability: float) -> float:
        clipped = min(1.0 - 1e-9, max(1e-9, float(probability)))
        logit = math.log(clipped / (1.0 - clipped))
        value = self.coefficient * logit + self.intercept
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def _fit_platt(rows, method):
    from sklearn.linear_model import LogisticRegression

    output = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        task_rows = [row for row in rows if row["benchmark"] == benchmark]
        probabilities = [_row_probability(row, method) for row in task_rows]
        labels = [int(float(row["target_gain"]) > 0.0) for row in task_rows]
        if len(set(labels)) < 2:
            output[(benchmark, method)] = _Platt(1.0, 0.0)
            continue
        logits = [
            [math.log(_clip(value) / (1.0 - _clip(value)))]
            for value in probabilities
        ]
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(logits, labels)
        output[(benchmark, method)] = _Platt(
            float(model.coef_[0][0]),
            float(model.intercept_[0]),
        )
    return output


def _fit_gain_conformal(rows, method):
    output = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        task_rows = [row for row in rows if row["benchmark"] == benchmark]
        observations = [
            ConformalGainObservation(
                record_id=row["record_id"],
                predicted_mean_gain=_row_expected_gain(row, method),
                predicted_scale=1.0,
                observed_gain=float(row["target_gain"]),
            )
            for row in task_rows
        ]
        output[(benchmark, method)] = fit_split_conformal_gain(
            observations,
            alpha=0.10,
        )
    return output


def _evaluate_scores(
    rows,
    scores,
    *,
    budget_fraction,
    positive_threshold,
    alpha,
):
    budget = max(1, math.ceil(budget_fraction * len(rows)))
    eligible = [
        index for index, score in enumerate(scores) if float(score) > positive_threshold
    ]
    eligible.sort(key=lambda index: (-float(scores[index]), rows[index]["record_id"]))
    selected = set(eligible[:budget])
    trajectory_values = []
    selected_gains = []
    task = {}
    for index, row in enumerate(rows):
        evidence_cost = float(row["acquired_evidence_cost"])
        if index in selected:
            value = float(row["target_gain"])
            selected_gains.append(value)
        else:
            value = -COST_WEIGHT * evidence_cost
        trajectory_values.append(value)
        bucket = task.setdefault(
            row["benchmark"],
            {"count": 0, "selected": 0, "errors": 0, "total_gain": 0.0},
        )
        bucket["count"] += 1
        bucket["total_gain"] += value
        if index in selected:
            bucket["selected"] += 1
            bucket["errors"] += int(value <= 0.0)
    errors = sum(value <= 0.0 for value in selected_gains)
    count = len(selected_gains)
    for bucket in task.values():
        bucket["coverage"] = bucket["selected"] / bucket["count"]
        bucket["selective_risk"] = (
            bucket["errors"] / bucket["selected"] if bucket["selected"] else 0.0
        )
    upper = binomial_upper_confidence(errors, count, 0.05) if count else 1.0
    return {
        "selected_record_ids": sorted(rows[index]["record_id"] for index in selected),
        "selected_count": count,
        "coverage": count / len(rows),
        "errors": errors,
        "selective_risk": errors / count if count else 0.0,
        "risk_upper_bound": upper,
        "certified": bool(count and upper <= alpha),
        "selected_total_strict_gain": sum(selected_gains),
        "selected_mean_strict_gain": (
            sum(selected_gains) / count if count else 0.0
        ),
        "trajectory_total_strict_gain": sum(trajectory_values),
        "population_mean_strict_gain": sum(trajectory_values) / len(rows),
        "task_breakdown": task,
    }


def _row_probability(row, method):
    if method == "ensemble":
        return float(row["mean_p_positive_gain"])
    return float(row["models"][method]["p_positive_gain"])


def _row_expected_gain(row, method):
    if method == "ensemble":
        return float(row["mean_expected_gain"])
    return float(row["models"][method]["expected_gain"])


def _clip(value):
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _parse_stage_paths(values):
    output = {}
    for value in values:
        stage, separator, path = value.partition("=")
        if not separator:
            raise ValueError("stage paths must use STAGE=PATH")
        output[stage] = json.loads(Path(path).read_text(encoding="utf-8"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", action="append", required=True)
    parser.add_argument("--acceptance", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    feedback = _parse_stage_paths(args.feedback)
    acceptance = _parse_stage_paths(args.acceptance)
    if set(feedback) != set(acceptance):
        raise ValueError("feedback and acceptance stages must match")
    payload = evaluate_same_evidence_judges(
        feedback,
        acceptance,
        budget_fraction=args.budget_fraction,
        alpha=args.alpha,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["strongest_result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
