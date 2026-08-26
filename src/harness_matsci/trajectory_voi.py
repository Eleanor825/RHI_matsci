from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .features import feature_names_from_records
from .historical import MAIN_MATERIAL_TASKS, load_historical_task_records
from .training import split_records
from .voi import VOI_SEED_HARNESS, fit_model_from_json, fit_voi_model, propose_voi_harness, summarize_voi_feedback, train_voi_rhi
from .schema import ActionRecord
from .trajectory_baselines import _metrics, _rows, _record, _top_fraction


def _split_training_pool(records: list[ActionRecord], seed: int) -> tuple[list[ActionRecord], list[ActionRecord], list[ActionRecord]]:
    train, feedback, acceptance = split_records(records, seed=seed, train_fraction=0.65, val_fraction=0.20)
    if not train or not feedback or not acceptance:
        raise ValueError("training pool must produce non-empty train/feedback/acceptance splits")
    return train, feedback, acceptance


def _sampled_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {task: [row for row in rows if row["task"] == task] for task in MAIN_MATERIAL_TASKS}


def _fit_models_for_task(
    all_records: list[ActionRecord],
    sampled_rows: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int,
    iterations: int,
) -> dict[str, Any]:
    sampled_ids = {row["source_record_id"] for row in sampled_rows}
    sampled_records = [record for record in all_records if record.record_id in sampled_ids]
    pool = [record for record in all_records if record.record_id not in sampled_ids]
    train, feedback, acceptance = _split_training_pool(pool, seed)
    available = feature_names_from_records(train + feedback)
    feedback_summary = summarize_voi_feedback(
        feedback,
        fit_voi_model(train, feedback, VOI_SEED_HARNESS, alpha=0.1, epochs=epochs, learning_rate=0.08, l2=0.01),
        budget_fraction=0.1,
    )
    static_harness, static_rationale = propose_voi_harness(
        VOI_SEED_HARNESS,
        feedback_summary,
        iteration=1,
        component="full",
        available_features=available,
    )
    static_model = fit_voi_model(
        train,
        feedback,
        static_harness,
        alpha=0.1,
        epochs=epochs,
        learning_rate=0.08,
        l2=0.01,
    )
    rhi = train_voi_rhi(
        train,
        feedback,
        acceptance,
        sampled_records,
        iterations=iterations,
        seed=seed,
        alpha=0.1,
        budget_fraction=0.1,
        epochs=epochs,
        learning_rate=0.08,
        l2=0.01,
        acceptance_policy="robust_guarded",
    )
    rhi_model = fit_model_from_json(rhi["final_model"])
    return {
        "sampled_records": sampled_records,
        "static_model": static_model,
        "static_harness": static_harness,
        "static_rationale": static_rationale,
        "rhi_model": rhi_model,
        "rhi_report": rhi,
        "training_sizes": {"train": len(train), "feedback": len(feedback), "acceptance": len(acceptance), "test": len(sampled_records)},
    }


def evaluate_trajectory_voi(
    trajectory_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 1729,
    epochs: int = 45,
    iterations: int = 2,
    budget_fraction: float = 0.10,
) -> dict[str, Any]:
    rows = _rows(trajectory_path)
    rows_by_task = _sampled_by_task(rows)
    outputs: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    selections: dict[str, dict[str, set[str]]] = {}
    for task_index, task in enumerate(MAIN_MATERIAL_TASKS):
        all_records = load_historical_task_records(data_dir, task)
        fitted = _fit_models_for_task(
            all_records,
            rows_by_task[task],
            seed=seed + task_index,
            epochs=epochs,
            iterations=iterations,
        )
        for method, model in [("static_sci_voi", fitted["static_model"]), ("scivoi_rhi", fitted["rhi_model"])]:
            task_rows = rows_by_task[task]
            task_records = fitted["sampled_records"]
            predictions = model.predict(task_records)
            actual_selected = {index for index, prediction in enumerate(predictions) if prediction.decision == "execute"}
            top_selected = _top_fraction([prediction.action_score for prediction in predictions], budget_fraction)
            for policy_name, selected in [("runtime_route", actual_selected), ("top10_budget", top_selected)]:
                result = _metrics(task_rows, selected)
                outputs.setdefault(method, {}).setdefault(policy_name, {})[task] = result
                selections.setdefault(method, {}).setdefault(policy_name, {})[task] = {
                    task_rows[index]["source_record_id"] for index in selected
                }
            for index, prediction in enumerate(predictions):
                decisions.append({
                    "task": task,
                    "method": method,
                    "record_id": task_records[index].record_id,
                    "prediction": prediction.to_json(),
                })
        outputs.setdefault("training_sizes", {})[task] = fitted["training_sizes"]
        outputs.setdefault("rhi_acceptance", {})[task] = {
            "accepted_versions": fitted["rhi_report"]["accepted_versions"],
            "history": fitted["rhi_report"]["history"],
        }
    for method in ["static_sci_voi", "scivoi_rhi"]:
        for policy_name in ["runtime_route", "top10_budget"]:
            task_results = outputs[method][policy_name]
            combined = []
            for task in MAIN_MATERIAL_TASKS:
                combined.extend([row for row in rows_by_task[task]])
            selected_ids = set().union(*(selections[method][policy_name][task] for task in MAIN_MATERIAL_TASKS))
            selected = {index for index, row in enumerate(combined) if row["source_record_id"] in selected_ids}
            outputs[method][policy_name]["overall"] = _metrics(combined, selected)
    report = {
        "schema_version": "trajectory-sci-voi-evaluation-v1",
        "trajectory_path": str(trajectory_path),
        "data_dir": str(data_dir),
        "n_trajectories": len(rows),
        "seed": seed,
        "epochs": epochs,
        "iterations": iterations,
        "budget_fraction": budget_fraction,
        "results": outputs,
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "scivoi_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (destination / "policy_decisions.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in decisions) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sci-VoI and Sci-VoI-RHI on fixed prompt trajectories.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    report = evaluate_trajectory_voi(args.trajectories, args.data_dir, args.out_dir, seed=args.seed, epochs=args.epochs, iterations=args.iterations)
    for method in ["static_sci_voi", "scivoi_rhi"]:
        print(method, json.dumps(report["results"][method], ensure_ascii=False))


if __name__ == "__main__":
    main()
