from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from .historical import MAIN_MATERIAL_TASKS, grouped_four_way_split, load_historical_tasks
from .llm_worthiness import LLMActionWorthinessClient
from .metrics import binary_metrics
from .replica_contract import ActionReplicaGrid, AgentEvidencePrediction, EvidenceView
from .scientific_gain import BudgetReservationValues, EmpiricalUtilityNormalizer, GainConfig, gain_targets
from .sequential_tool_benchmark import SequentialToolEnvironment


DEFAULT_MODELS = ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra")


def run_closed_llm_sequential_pilot(
    raw_data_dir: str | Path,
    out_dir: str | Path,
    *,
    api_key: str,
    base_url: str,
    models: tuple[str, ...] = DEFAULT_MODELS,
    split_seed: int = 7,
    n_per_task: int = 10,
    workers: int = 12,
    selection_mode: str = "deterministic",
) -> dict[str, Any]:
    records_by_task = load_historical_tasks(raw_data_dir, MAIN_MATERIAL_TASKS)
    splits_by_task = {
        task: grouped_four_way_split(records, seed=split_seed)
        for task, records in records_by_task.items()
    }
    train = [record for task in MAIN_MATERIAL_TASKS for record in splits_by_task[task]["train"]]
    test = [record for task in MAIN_MATERIAL_TASKS for record in splits_by_task[task]["test"]]
    normalizer = EmpiricalUtilityNormalizer.fit(train)
    config = GainConfig()
    reservations = BudgetReservationValues.fit(
        train,
        normalizer,
        config,
        budget_fraction=0.10,
        outside_option=0.0,
    )
    test_targets = gain_targets(test, normalizer, config, reservations)
    test_target_by_id = {target.record_id: target for target in test_targets}
    selected = []
    for task in MAIN_MATERIAL_TASKS:
        task_records = [record for record in test if record.benchmark == task]
        task_records.sort(key=lambda record: _rank(split_seed, record.record_id))
        if selection_mode == "deterministic":
            selected.extend(task_records[:n_per_task])
        elif selection_mode == "label_balanced_diagnostic":
            positive = [record for record in task_records if test_target_by_id[record.record_id].worthy == 1]
            negative = [record for record in task_records if test_target_by_id[record.record_id].worthy == 0]
            per_label = n_per_task // 2
            if n_per_task % 2 or len(positive) < per_label or len(negative) < per_label:
                raise ValueError("balanced diagnostic requires an even n_per_task and both worthiness labels")
            selected.extend(positive[:per_label])
            selected.extend(negative[:per_label])
        else:
            raise ValueError(f"unknown selection_mode {selection_mode!r}")
    target_by_id = {record.record_id: test_target_by_id[record.record_id] for record in selected}
    environment = SequentialToolEnvironment.fit(raw_data_dir, train)
    views_by_id = {
        record.record_id: [
            _evidence_view(
                environment.observe(
                    record,
                    stage="verification",
                    evidence_replica=replica,
                    run_seed=split_seed + 9000,
                ),
                replica,
            )
            for replica in (101, 211, 307)
        ]
        for record in selected
    }
    destination = Path(out_dir)
    cache_dir = destination / "cache"
    clients = {
        model: LLMActionWorthinessClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            cache_path=cache_dir / f"{model.replace('/', '__')}.json",
            timeout=180.0,
            max_retries=3,
        )
        for model in models
    }
    jobs = [
        (record, view, model)
        for record in selected
        for view in views_by_id[record.record_id]
        for model in models
    ]
    predictions: dict[tuple[str, str, str], Any] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                clients[model].predict,
                view,
                reservation_value=reservations.value(record.benchmark),
                cost_weight=config.cost_weight,
                failure_harm_weight=config.failure_harm_weight,
            ): (record.record_id, view.view_id, model)
            for record, view, model in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                predictions[key] = future.result()
            except Exception as error:
                failures.append(
                    {
                        "record_id": key[0],
                        "view_id": key[1],
                        "model": key[2],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    destination.mkdir(parents=True, exist_ok=True)
    if failures:
        (destination / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"closed-model sequential pilot has {len(failures)} failed cells")
    stale_failures = destination / "failures.json"
    if stale_failures.exists():
        stale_failures.unlink()
    action_rows = []
    prediction_rows = []
    for record in selected:
        cells = []
        for model in models:
            for view in views_by_id[record.record_id]:
                prediction = predictions[(record.record_id, view.view_id, model)]
                cells.append(
                    AgentEvidencePrediction(
                        record_id=record.record_id,
                        agent_replica_id=model,
                        evidence_view_id=view.view_id,
                        predicted_gain=prediction.expected_gain,
                        positive_gain_probability=prediction.p_positive_gain,
                        backend="closed_model_responses_api",
                        internal_signals={
                            "assessment_confidence": prediction.assessment_confidence,
                            "p_success": prediction.p_success,
                        },
                        external_signals={
                            "tool_uncertainty": float(view.record.features["tool_uncertainty"]),
                            "tool_positive_probability": float(
                                view.record.features["tool_positive_probability"]
                            ),
                            "tool_fidelity": float(view.record.features["tool_fidelity"]),
                        },
                    )
                )
                prediction_rows.append(
                    {
                        "record_id": record.record_id,
                        "benchmark": record.benchmark,
                        "model": model,
                        "view_id": view.view_id,
                        **prediction.to_json(),
                        "view_provenance": view.provenance,
                    }
                )
        grid = ActionReplicaGrid.from_predictions(cells)
        grid_summary = grid.summarize()
        target = target_by_id[record.record_id]
        action_rows.append(
            {
                "record_id": record.record_id,
                "benchmark": record.benchmark,
                "group_id": str(record.metadata.get("group_id", record.benchmark)),
                "worthy": target.worthy,
                "target_gain": target.gain,
                **grid_summary.to_json(),
            }
        )
    report = {
        "schema_version": "sd-worth-h6-closed-llm-sequential-v2",
        "models": list(models),
        "split_seed": split_seed,
        "n_per_task": n_per_task,
        "selection_mode": selection_mode,
        "n_actions": len(action_rows),
        "n_prediction_cells": len(prediction_rows),
        "matrix_complete": len(prediction_rows) == len(action_rows) * len(models) * 3,
        "evidence_view_ids": [view.view_id for view in views_by_id[selected[0].record_id]],
        "summary": _summarize(action_rows, prediction_rows, models),
    }
    with (destination / "action_summaries.jsonl").open("w", encoding="utf-8") as handle:
        for row in action_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (destination / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (destination / "RESULTS.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _evidence_view(observation, replica: int) -> EvidenceView:
    return EvidenceView(
        view_id=f"tool::verification::replica-{replica}",
        record=observation.record,
        intervention="independent_verification_observation",
        provenance=observation.provenance,
    )


def _summarize(action_rows, prediction_rows, models):
    labels = [row["worthy"] for row in action_rows]
    probabilities = [row["mean_positive_gain_probability"] for row in action_rows]
    metrics = binary_metrics(labels, probabilities, 0.5)
    errors = [abs(row["mean_predicted_gain"] - row["target_gain"]) for row in action_rows]
    internal = [row["source_variance"]["internal_variance"] for row in action_rows]
    external = [row["source_variance"]["external_variance"] for row in action_rows]
    return {
        "label_counts": {
            "negative": labels.count(0),
            "positive": labels.count(1),
        },
        "discrimination_metrics_valid": len(set(labels)) == 2,
        "ensemble": {
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "aurc": metrics["aurc"],
            "accuracy": metrics["accuracy_at_threshold"],
            "expected_gain_mae": mean(errors),
        },
        "source_components": {
            "mean_internal_variance": mean(internal),
            "mean_external_variance": mean(external),
            "internal_variance_error_correlation": _correlation(internal, errors),
            "external_variance_error_correlation": _correlation(external, errors),
        },
        "by_task": {
            task: _subset_summary([row for row in action_rows if row["benchmark"] == task])
            for task in MAIN_MATERIAL_TASKS
        },
        "by_model": {
            model: _model_summary(action_rows, prediction_rows, model)
            for model in models
        },
    }


def _subset_summary(rows):
    errors = [abs(row["mean_predicted_gain"] - row["target_gain"]) for row in rows]
    return {
        "n": len(rows),
        "mean_internal_variance": mean(row["source_variance"]["internal_variance"] for row in rows),
        "mean_external_variance": mean(row["source_variance"]["external_variance"] for row in rows),
        "expected_gain_mae": mean(errors),
    }


def _model_summary(action_rows, prediction_rows, model):
    rows = [row for row in prediction_rows if row["model"] == model]
    by_id = {}
    for row in rows:
        by_id.setdefault(row["record_id"], []).append(row)
    probabilities = [mean(row["p_positive_gain"] for row in by_id[action["record_id"]]) for action in action_rows]
    labels = [action["worthy"] for action in action_rows]
    metrics = binary_metrics(labels, probabilities, 0.5)
    return {"brier": metrics["brier"], "ece": metrics["ece"], "aurc": metrics["aurc"]}


def _correlation(left, right):
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = (
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def _rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|h6|{record_id}".encode()).hexdigest()


def _markdown(report):
    ensemble = report["summary"]["ensemble"]
    source = report["summary"]["source_components"]
    return "\n".join(
        [
            "# H6 Closed-Model Sequential-v2 Pilot",
            "",
            f"- Complete matrix: `{report['matrix_complete']}`",
            f"- Actions: `{report['n_actions']}`",
            f"- Prediction cells: `{report['n_prediction_cells']}`",
            f"- Selection mode: `{report['selection_mode']}`",
            f"- Label counts: `{json.dumps(report['summary']['label_counts'], sort_keys=True)}`",
            f"- Discrimination metrics valid: `{report['summary']['discrimination_metrics_valid']}`",
            f"- Ensemble Brier: `{ensemble['brier']:.4f}`",
            f"- Ensemble ECE: `{ensemble['ece']:.4f}`",
            f"- Ensemble AURC: `{ensemble['aurc']:.4f}`",
            f"- Expected-gain MAE: `{ensemble['expected_gain_mae']:.4f}`",
            f"- Mean internal variance: `{source['mean_internal_variance']:.6f}`",
            f"- Mean external variance: `{source['mean_external_variance']:.6f}`",
            f"- Corr(internal variance, gain error): `{source['internal_variance_error_correlation']:.4f}`",
            f"- Corr(external variance, gain error): `{source['external_variance_error_correlation']:.4f}`",
            "",
        ]
    )
