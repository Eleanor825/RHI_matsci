from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from harness_matsci.decision_instability import decision_boundary_instability
from harness_matsci.metrics import binary_metrics, fixed_coverage_metrics


METHODS = (
    "llm_direct_probability",
    "self_report_shrinkage",
    "action_distribution_harness",
    "action_distribution_plus_model",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    result = evaluate(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


def evaluate(payload):
    records = _aggregate_records(payload["rows"])
    batches = sorted({row["source_batch"] for row in records})
    predictions = {method: {} for method in METHODS}
    coefficients = {method: [] for method in METHODS if method != METHODS[0]}
    for held_batch in batches:
        train = [row for row in records if row["source_batch"] != held_batch]
        held = [row for row in records if row["source_batch"] == held_batch]
        for row in held:
            predictions["llm_direct_probability"][row["record_id"]] = row["p_raw"]
        for method in METHODS[1:]:
            feature_names = _feature_names(method)
            model = _fit_logistic(train, feature_names)
            coefficients[method].append(
                {
                    "held_batch": held_batch,
                    "feature_names": feature_names,
                    "intercept": float(model.intercept_[0]),
                    "coefficients": [float(value) for value in model.coef_[0]],
                }
            )
            values = model.predict_proba(
                [_features(row, feature_names) for row in held]
            )[:, 1]
            for row, value in zip(held, values):
                predictions[method][row["record_id"]] = float(value)
    labels = [row["label"] for row in records]
    ordered_ids = [row["record_id"] for row in records]
    summary = {}
    for method in METHODS:
        probabilities = [predictions[method][record_id] for record_id in ordered_ids]
        summary[method] = {
            "metrics": binary_metrics(labels, probabilities, threshold=0.5),
            "fixed_coverage": fixed_coverage_metrics(
                labels,
                probabilities,
                coverages=(0.10, 0.25, 0.50),
            ),
        }
    summary["action_entropy_error_diagnostic"] = {
        "correlation": _correlation(
            [row["action_entropy"] for row in records],
            [float((row["p_raw"] >= 0.5) != bool(row["label"])) for row in records],
        ),
        "mean_entropy": _mean(row["action_entropy"] for row in records),
        "nonzero_fraction": _mean(float(row["action_entropy"] > 0.0) for row in records),
    }
    return {
        "development_only": True,
        "protocol": "leave-one-source-batch-out",
        "record_count": len(records),
        "batches": batches,
        "summary": summary,
        "coefficients": coefficients,
        "records": records,
        "out_of_batch_predictions": predictions,
    }


def _aggregate_records(rows):
    by_id = {}
    for row in rows:
        entry = by_id.setdefault(
            row["record_id"],
            {
                "record_id": row["record_id"],
                "source_batch": row["source_batch"],
                "label": row["label_a_better"],
                "replicas": [],
                "model_probabilities": [],
            },
        )
        entry["replicas"].extend(row["replicas"])
        entry["model_probabilities"].append(row["mean_p_a_better"])
    output = []
    for entry in sorted(by_id.values(), key=lambda value: value["record_id"]):
        choices = [
            1.0 if replica["choice"] == "A" else -1.0
            for replica in entry["replicas"]
        ]
        instability = decision_boundary_instability(choices)
        probabilities = [replica["p_a_better"] for replica in entry["replicas"]]
        p_raw = _mean(probabilities)
        self_uncertainty = _mean(
            1.0 - replica["assessment_confidence"] for replica in entry["replicas"]
        )
        model_mean = _mean(entry["model_probabilities"])
        model_variance = _mean(
            (value - model_mean) ** 2 for value in entry["model_probabilities"]
        )
        output.append(
            {
                "record_id": entry["record_id"],
                "source_batch": entry["source_batch"],
                "label": entry["label"],
                "p_raw": p_raw,
                "choice_fraction_a": instability.positive_fraction,
                "action_entropy": instability.normalized_vote_entropy,
                "self_reported_uncertainty": self_uncertainty,
                "model_probability_variance": model_variance,
            }
        )
    return output


def _feature_names(method):
    if method == "self_report_shrinkage":
        return ("logit", "logit_x_self")
    if method == "action_distribution_harness":
        return ("logit", "logit_x_action_entropy")
    if method == "action_distribution_plus_model":
        return ("logit", "logit_x_action_entropy", "logit_x_model_scale")
    raise ValueError(method)


def _features(row, feature_names):
    logit = _logit(row["p_raw"])
    values = {
        "logit": logit,
        "logit_x_self": logit * row["self_reported_uncertainty"],
        "logit_x_action_entropy": logit * row["action_entropy"],
        "logit_x_model_scale": logit * math.sqrt(row["model_probability_variance"]),
    }
    return [values[name] for name in feature_names]


def _fit_logistic(records, feature_names):
    from sklearn.linear_model import LogisticRegression

    labels = [row["label"] for row in records]
    if len(set(labels)) < 2:
        raise ValueError("training batches require both labels")
    return LogisticRegression(C=0.25, solver="lbfgs", max_iter=2000).fit(
        [_features(row, feature_names) for row in records],
        labels,
    )


def _logit(probability):
    probability = min(1.0 - 1e-5, max(1e-5, probability))
    return math.log(probability / (1.0 - probability))


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _correlation(left, right):
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


if __name__ == "__main__":
    main()
