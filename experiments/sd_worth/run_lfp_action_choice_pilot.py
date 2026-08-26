from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness_matsci.decision_instability import (
    DecisionBoundaryInstability,
    decision_boundary_instability,
)
from harness_matsci.llm_action_choice import LLMActionChoiceClient
from harness_matsci.metrics import binary_metrics
from harness_matsci.schema import ActionRecord


DEFAULT_MODELS = ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--per-batch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()
    models = tuple(value for value in args.models.split(",") if value)
    records = _load_records(args.records)
    selected = _select_outcome_blind(records, per_batch=args.per_batch)
    report = run_pilot(
        selected,
        output_dir=args.output_dir,
        models=models,
        replicas=args.replicas,
        workers=args.workers,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(report["summary"], sort_keys=True))


def run_pilot(
    records,
    *,
    output_dir,
    models,
    replicas,
    workers,
    reasoning_effort,
):
    if replicas < 1:
        raise ValueError("action-choice evaluation requires at least one replica")
    destination = Path(output_dir)
    cache_dir = destination / "cache"
    clients = {
        model: LLMActionChoiceClient.from_env(
            model=model,
            cache_path=cache_dir / f"{model}.json",
            reasoning_effort=reasoning_effort,
            timeout=180.0,
            max_retries=2,
        )
        for model in models
    }
    predictions = {}
    failures = []
    jobs = [
        (record, model, replica)
        for record in records
        for model in models
        for replica in range(replicas)
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                clients[model].predict,
                record,
                replica_id=replica,
            ): (record.record_id, model, replica)
            for record, model, replica in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                predictions[key] = future.result()
            except Exception as error:
                failures.append(
                    {
                        "record_id": key[0],
                        "model": key[1],
                        "replica": key[2],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    destination.mkdir(parents=True, exist_ok=True)
    if failures:
        (destination / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise RuntimeError(f"action-choice pilot has {len(failures)} failed cells")
    failure_path = destination / "failures.json"
    if failure_path.exists():
        failure_path.unlink()
    rows = []
    for record in records:
        for model in models:
            model_predictions = [
                predictions[(record.record_id, model, replica)]
                for replica in range(replicas)
            ]
            choices = [1.0 if row.choice == "A" else -1.0 for row in model_predictions]
            instability = _decision_instability(choices)
            rows.append(
                {
                    "record_id": record.record_id,
                    "source_batch": record.metadata["source_batch"],
                    "label_a_better": record.label,
                    "model": model,
                    "replicas": [row.to_json() for row in model_predictions],
                    "mean_p_a_better": _mean(row.p_a_better for row in model_predictions),
                    "majority_choice_a": int(
                        sum(row.choice == "A" for row in model_predictions)
                        > len(model_predictions) / 2.0
                    ),
                    "choice_positive_fraction": instability.positive_fraction,
                    "choice_entropy": instability.normalized_vote_entropy,
                    "choice_flip": bool(instability.normalized_vote_entropy > 0.0),
                    "mean_self_reported_uncertainty": _mean(
                        1.0 - row.assessment_confidence for row in model_predictions
                    ),
                }
            )
    summary = _summarize(rows, models)
    payload = {
        "development_only": True,
        "selection": "first records per source batch after record-id sort; outcomes unused",
        "record_count": len(records),
        "models": list(models),
        "replicas": replicas,
        "rows": rows,
        "summary": summary,
    }
    (destination / "predictions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    (destination / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _summarize(rows, models):
    output = {}
    for model in (*models, "all_models"):
        if model == "all_models":
            by_record = defaultdict(list)
            for row in rows:
                by_record[row["record_id"]].append(row)
            labels = [values[0]["label_a_better"] for values in by_record.values()]
            probabilities = [
                _mean(value["mean_p_a_better"] for value in values)
                for values in by_record.values()
            ]
            majority = [
                int(_mean(value["majority_choice_a"] for value in values) >= 0.5)
                for values in by_record.values()
            ]
            choice_entropy = [
                decision_boundary_instability(
                    [
                        1.0 if replica["choice"] == "A" else -1.0
                        for value in values
                        for replica in value["replicas"]
                    ]
                ).normalized_vote_entropy
                for values in by_record.values()
            ]
            self_uncertainty = [
                _mean(value["mean_self_reported_uncertainty"] for value in values)
                for values in by_record.values()
            ]
        else:
            selected = [row for row in rows if row["model"] == model]
            labels = [row["label_a_better"] for row in selected]
            probabilities = [row["mean_p_a_better"] for row in selected]
            majority = [row["majority_choice_a"] for row in selected]
            choice_entropy = [row["choice_entropy"] for row in selected]
            self_uncertainty = [row["mean_self_reported_uncertainty"] for row in selected]
        errors = [float(prediction != label) for prediction, label in zip(majority, labels)]
        output[model] = {
            "n": len(labels),
            "majority_accuracy": _mean(1.0 - error for error in errors),
            "choice_flip_fraction": _mean(float(value > 0.0) for value in choice_entropy),
            "mean_choice_entropy": _mean(choice_entropy),
            "mean_self_reported_uncertainty": _mean(self_uncertainty),
            "self_uncertainty_choice_entropy_correlation": _correlation(
                self_uncertainty, choice_entropy
            ),
            "self_uncertainty_error_correlation": _correlation(self_uncertainty, errors),
            "choice_entropy_error_correlation": _correlation(choice_entropy, errors),
            "calibration": binary_metrics(labels, probabilities, threshold=0.5),
        }
    return output


def _load_records(path):
    return [
        ActionRecord.from_json(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_outcome_blind(records, *, per_batch):
    by_batch = defaultdict(list)
    for record in records:
        by_batch[record.metadata["source_batch"]].append(record)
    return [
        record
        for batch in sorted(by_batch)
        for record in (
            sorted(by_batch[batch], key=lambda row: row.record_id)[:per_batch]
            if per_batch > 0
            else sorted(by_batch[batch], key=lambda row: row.record_id)
        )
    ]


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _decision_instability(values):
    rows = list(values)
    if len(rows) >= 2:
        return decision_boundary_instability(rows)
    positive_fraction = float(bool(rows and rows[0] > 0.0))
    return DecisionBoundaryInstability(
        positive_fraction=positive_fraction,
        normalized_vote_entropy=0.0,
        value_range=0.0,
        boundary_instability=0.0,
    )


def _correlation(left, right):
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


def _markdown(payload):
    lines = [
        "# LFP Historical Action-Choice Pilot",
        "",
        f"- Records: {payload['record_count']}",
        f"- Replicas per model: {payload['replicas']}",
        "- Selection is outcome-blind and development-only.",
        "",
        "| model | accuracy | Brier | ECE | flip fraction | choice entropy | self uncertainty | corr(self, entropy) | corr(entropy, error) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in payload["summary"].items():
        lines.append(
            f"| {model} | {values['majority_accuracy']:.3f} | "
            f"{values['calibration']['brier']:.3f} | {values['calibration']['ece']:.3f} | "
            f"{values['choice_flip_fraction']:.3f} | {values['mean_choice_entropy']:.3f} | "
            f"{values['mean_self_reported_uncertainty']:.3f} | "
            f"{values['self_uncertainty_choice_entropy_correlation']:.3f} | "
            f"{values['choice_entropy_error_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Self-reported uncertainty is not treated as internal variance. The empirical "
            "choice distribution across independent calls is the measured internal signal.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
