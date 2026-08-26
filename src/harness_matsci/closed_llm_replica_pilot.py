from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from .historical import MAIN_MATERIAL_TASKS, load_historical_tasks, split_historical_tasks
from .llm_worthiness import LLMActionWorthinessClient
from .metrics import binary_metrics
from .replica_contract import ActionReplicaGrid, AgentEvidencePrediction
from .retrieval_views import HistoricalEvidenceRetriever
from .scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    gain_targets,
)
from .sd_worth_experiment import _with_task_features


DEFAULT_MODELS = ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra")
SAMPLE_SEED = 260712447


def run_closed_llm_replica_pilot(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    split_seed: int = 7,
    n_per_task: int = 10,
    budget_fraction: float = 0.10,
    workers: int = 12,
    evidence_mode: str = "historical",
) -> dict[str, Any]:
    if evidence_mode not in {"historical", "candidate_simulator"}:
        raise ValueError("evidence_mode must be historical or candidate_simulator")
    records_by_task = load_historical_tasks(data_dir, MAIN_MATERIAL_TASKS)
    task_splits = split_historical_tasks(records_by_task, seed=split_seed)
    train = _with_task_features(
        [record for task in MAIN_MATERIAL_TASKS for record in task_splits[task]["train"]]
    )
    feedback = _with_task_features(
        [record for task in MAIN_MATERIAL_TASKS for record in task_splits[task]["feedback"]]
    )
    test_by_task = {
        task: _with_task_features(task_splits[task]["test"])
        for task in MAIN_MATERIAL_TASKS
    }
    selected = []
    rng = random.Random(SAMPLE_SEED)
    for task in MAIN_MATERIAL_TASKS:
        candidates = list(test_by_task[task])
        rng.shuffle(candidates)
        selected.extend(candidates[:n_per_task])

    config = GainConfig()
    normalizer = EmpiricalUtilityNormalizer.fit(train)
    reservation = BudgetReservationValues.fit(
        train, normalizer, config, budget_fraction=budget_fraction
    )
    targets = gain_targets(selected, normalizer, config, reservation)
    train_targets = gain_targets(train, normalizer, config, reservation)
    feedback_targets = gain_targets(feedback, normalizer, config, reservation)
    target_by_id = {target.record_id: target for target in targets}
    retriever = HistoricalEvidenceRetriever(train, neighbors=5)
    if evidence_mode == "candidate_simulator":
        from .simulator_views import CandidateSimulatorTool

        simulator = CandidateSimulatorTool(
            train, train_targets, feedback, feedback_targets, seed=SAMPLE_SEED
        )
        views_by_id = {}
        for record in selected:
            historical_views = retriever.evidence_views(record)
            views_by_id[record.record_id] = [
                historical_views[0], historical_views[1], simulator.evidence_view(record)
            ]
    else:
        views_by_id = {record.record_id: retriever.evidence_views(record) for record in selected}
    evidence_view_ids = [view.view_id for view in views_by_id[selected[0].record_id]]

    destination = Path(out_dir)
    cache_dir = destination / "cache"
    clients = {
        model: LLMActionWorthinessClient.from_env(
            model=model,
            cache_path=cache_dir / f"{model.replace('/', '__')}.json",
            timeout=180.0,
            max_retries=2,
        )
        for model in models
    }
    jobs = []
    for record in selected:
        for view in views_by_id[record.record_id]:
            for model in models:
                jobs.append((record.record_id, view, model))
    predictions: dict[tuple[str, str, str], Any] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                clients[model].predict,
                view,
                reservation_value=reservation.value(view.record.benchmark),
                cost_weight=config.cost_weight,
                failure_harm_weight=config.failure_harm_weight,
            ): (record_id, view.view_id, model)
            for record_id, view, model in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                predictions[key] = future.result()
            except Exception as error:
                failures.append({
                    "record_id": key[0], "view_id": key[1], "model": key[2],
                    "error": f"{type(error).__name__}: {error}",
                })
    if failures:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"closed-model replica pilot has {len(failures)} failed cells")
    transient_failures = destination / "failures.json"
    if transient_failures.exists():
        transient_failures.unlink()

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
                            key: float(view.record.features.get(key, 0.0))
                            for key in (
                                "evidence_support", "evidence_conflict", "source_reliability",
                                "tool_agreement", "ood_score", "source_risk",
                            )
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
        summary = grid.summarize()
        target = target_by_id[record.record_id]
        base_cells = [cell for cell in cells if cell.evidence_view_id == "tool::base_evidence"]
        view_means = {
            view_id: mean(cell.predicted_gain for cell in cells if cell.evidence_view_id == view_id)
            for view_id in grid.evidence_view_ids
        }
        tool_gain_shifts = {
            view_id: view_means[view_id] - view_means["tool::base_evidence"]
            for view_id in grid.evidence_view_ids
            if view_id != "tool::base_evidence"
        }
        action_rows.append(
            {
                "record_id": record.record_id,
                "benchmark": record.benchmark,
                "group_id": str(record.metadata.get("group_id", record.benchmark)),
                "worthy": target.worthy,
                "target_gain": target.gain,
                "base_probability": mean(cell.positive_gain_probability for cell in base_cells),
                "all_view_probability": summary.mean_positive_gain_probability,
                "base_expected_gain": mean(cell.predicted_gain for cell in base_cells),
                **summary.to_json(),
                "tool_gain_shifts": tool_gain_shifts,
            }
        )

    report = {
        "schema_version": "sd-worth-h2.1-closed-llm-pilot-v1",
        "protocol": "research/evl_iclr/experiments/h2_variance_decomposition/protocol_closed_llm_pilot.md",
        "models": list(models),
        "split_seed": split_seed,
        "sample_seed": SAMPLE_SEED,
        "n_per_task": n_per_task,
        "n_actions": len(action_rows),
        "n_prediction_cells": len(prediction_rows),
        "matrix_complete": len(prediction_rows) == len(action_rows) * len(models) * len(evidence_view_ids),
        "evidence_mode": evidence_mode,
        "evidence_view_ids": evidence_view_ids,
        "reservation_values": reservation.to_json(),
        "summary": _summarize(action_rows, prediction_rows, models, evidence_view_ids),
    }
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "action_summaries.jsonl").open("w") as handle:
        for row in action_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (destination / "predictions.jsonl").open("w") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (destination / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (destination / "RESULTS.md").write_text(_markdown(report))
    return report


def _summarize(action_rows, prediction_rows, models, evidence_view_ids):
    labels = [row["worthy"] for row in action_rows]
    base = [row["base_probability"] for row in action_rows]
    all_views = [row["all_view_probability"] for row in action_rows]
    output = {
        "base_ensemble": _probability_metrics(labels, base),
        "all_tool_views_ensemble": _probability_metrics(labels, all_views),
        "source_components": _source_summary(action_rows, evidence_view_ids),
        "by_task": {
            task: _source_summary(
                [row for row in action_rows if row["benchmark"] == task], evidence_view_ids
            )
            for task in MAIN_MATERIAL_TASKS
        },
        "by_model": {},
    }
    for model in models:
        base_rows = [
            row for row in prediction_rows
            if row["model"] == model and row["view_id"] == "tool::base_evidence"
        ]
        by_id = {row["record_id"]: row["p_positive_gain"] for row in base_rows}
        output["by_model"][model] = _probability_metrics(
            labels, [by_id[row["record_id"]] for row in action_rows]
        )
    return output


def _probability_metrics(labels, probabilities):
    metrics = binary_metrics(labels, probabilities, 0.5)
    return {
        "accuracy": metrics["accuracy_at_threshold"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "aurc": metrics["aurc"],
    }


def _source_summary(rows, evidence_view_ids):
    if not rows:
        return {"n": 0}
    internal = [row["source_variance"]["internal_variance"] for row in rows]
    external = [row["source_variance"]["external_variance"] for row in rows]
    errors = [abs(row["mean_predicted_gain"] - row["target_gain"]) for row in rows]
    result = {
        "n": len(rows),
        "mean_internal_variance": mean(internal),
        "mean_external_variance": mean(external),
        "mean_total_variance": mean(row["source_variance"]["total_variance"] for row in rows),
        "internal_variance_error_correlation": _correlation(internal, errors),
        "external_variance_error_correlation": _correlation(external, errors),
    }
    result["mean_tool_gain_shifts"] = {
        view_id: mean(row["tool_gain_shifts"].get(view_id, 0.0) for row in rows)
        for view_id in evidence_view_ids
        if view_id != "tool::base_evidence"
    }
    return result


def _correlation(left, right):
    left_mean = mean(left); right_mean = mean(right)
    numerator = sum((a-left_mean)*(b-right_mean) for a,b in zip(left,right))
    scale = (sum((a-left_mean)**2 for a in left) * sum((b-right_mean)**2 for b in right)) ** 0.5
    return numerator / scale if scale > 0 else 0.0


def _markdown(report):
    summary = report["summary"]
    lines = [
        "# H2.1 Closed-Model × Retrieval-Tool Pilot",
        "",
        f"Complete matrix: `{report['matrix_complete']}`; actions: `{report['n_actions']}`; cells: `{report['n_prediction_cells']}`.",
        "",
        "## Probability quality",
        "",
        "| Score | Brier | ECE | AURC | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("base_ensemble", "all_tool_views_ensemble"):
        values = summary[name]
        lines.append(
            f"| {name} | {values['brier']:.4f} | {values['ece']:.4f} | "
            f"{values['aurc']:.4f} | {values['accuracy']:.4f} |"
        )
    lines.extend([
        "", "## Source components", "",
        "| Scope | N | Internal var | External var | Corr internal/error | Corr external/error | Tool shifts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    groups = {"overall": summary["source_components"], **summary["by_task"]}
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['n']} | {values.get('mean_internal_variance',0):.5f} | "
            f"{values.get('mean_external_variance',0):.5f} | "
            f"{values.get('internal_variance_error_correlation',0):.4f} | "
            f"{values.get('external_variance_error_correlation',0):.4f} | "
            f"`{json.dumps(values.get('mean_tool_gain_shifts', {}), sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run real closed-model x retrieval-tool replica pilot.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--n-per-task", type=int, default=10)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--evidence-mode",
        choices=("historical", "candidate_simulator"),
        default="historical",
    )
    args = parser.parse_args()
    report = run_closed_llm_replica_pilot(
        args.data_dir, args.out_dir,
        models=tuple(value for value in args.models.split(",") if value),
        split_seed=args.split_seed, n_per_task=args.n_per_task,
        budget_fraction=args.budget_fraction, workers=args.workers,
        evidence_mode=args.evidence_mode,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
