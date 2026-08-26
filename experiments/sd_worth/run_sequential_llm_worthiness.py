from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness_matsci.llm_worthiness import LLMActionWorthinessClient
from harness_matsci.metrics import binary_metrics
from harness_matsci.replica_contract import EvidenceView
from harness_matsci.schema import ActionRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--stage", default="base")
    parser.add_argument("--cost-aware", action="store_true")
    parser.add_argument(
        "--models",
        default="gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--gain-definition",
        choices=("budget_conditioned", "strict_realized"),
        default="budget_conditioned",
    )
    args = parser.parse_args()
    trajectories = [
        json.loads(line)
        for line in Path(args.trajectories).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in trajectories if row["split"] == args.split]
    report = run_baseline(
        selected,
        output_dir=args.output_dir,
        stage=args.stage,
        models=tuple(value for value in args.models.split(",") if value),
        workers=args.workers,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        max_retries=args.max_retries,
        gain_definition=args.gain_definition,
        cost_aware=args.cost_aware,
    )
    print(json.dumps(report["summary"], sort_keys=True))


def run_baseline(
    trajectories,
    *,
    output_dir,
    stage,
    models,
    workers,
    reasoning_effort,
    timeout=180.0,
    max_retries=3,
    gain_definition="budget_conditioned",
    cost_aware=False,
):
    if cost_aware and gain_definition != "strict_realized":
        raise ValueError("cost-aware evidence views require strict_realized gain")
    destination = Path(output_dir)
    cache_dir = destination / "cache"
    reservation_values = _reservation_values(trajectories)
    views = {
        row["record_id"]: _view(row, stage, cost_aware=cost_aware)
        for row in trajectories
    }
    clients = {
        model: LLMActionWorthinessClient.from_env(
            model=model,
            cache_path=cache_dir / f"{model.replace('/', '__')}.json",
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_retries=max_retries,
            gain_definition=gain_definition,
        )
        for model in models
    }
    jobs = [(row, model) for row in trajectories for model in models]
    predictions = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                clients[model].predict,
                views[row["record_id"]],
                reservation_value=(
                    0.0
                    if gain_definition == "strict_realized"
                    else reservation_values[row["benchmark"]]
                ),
                cost_weight=0.15,
                failure_harm_weight=0.25,
            ): (row["record_id"], model)
            for row, model in jobs
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
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    for client in clients.values():
        client.flush_cache()
    destination.mkdir(parents=True, exist_ok=True)
    if failures:
        (destination / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise RuntimeError(f"sequential LLM baseline has {len(failures)} failed cells")
    rows = []
    for trajectory in trajectories:
        cells = [predictions[(trajectory["record_id"], model)] for model in models]
        target = trajectory["hidden_outcome_for_evaluation_only"]
        target_gain = float(
            target["base_gain"]
            if gain_definition == "strict_realized"
            else target["budget_conditioned_gain"]
        )
        if cost_aware:
            target_gain -= 0.15 * _acquired_cost(trajectory, stage)
        rows.append(
            {
                "record_id": trajectory["record_id"],
                "benchmark": trajectory["benchmark"],
                "split": trajectory["split"],
                "stage": stage,
                "cost_aware": cost_aware,
                "acquired_evidence_cost": _acquired_cost(trajectory, stage),
                "worthy": int(target_gain > 0.0),
                "target_gain": target_gain,
                "models": {
                    model: predictions[(trajectory["record_id"], model)].to_json()
                    for model in models
                },
                "mean_p_positive_gain": _mean(cell.p_positive_gain for cell in cells),
                "mean_expected_gain": _mean(cell.expected_gain for cell in cells),
                "probability_variance": _variance(cell.p_positive_gain for cell in cells),
                "expected_gain_variance": _variance(cell.expected_gain for cell in cells),
            }
        )
    report = {
        "schema_version": "sequential-v2-generic-llm-worthiness-v3",
        "gain_definition": gain_definition,
        "stage": stage,
        "cost_aware": cost_aware,
        "split": trajectories[0]["split"] if trajectories else "empty",
        "models": list(models),
        "record_count": len(rows),
        "cell_count": len(rows) * len(models),
        "reservation_values": reservation_values,
        "rows": rows,
        "summary": _summary(rows, models),
    }
    (destination / "predictions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _view(trajectory, stage: str, *, cost_aware: bool = False) -> EvidenceView:
    stage_names = tuple(value for value in stage.split("+") if value)
    if not stage_names:
        raise ValueError("stage bundle cannot be empty")
    rows_by_stage = {str(row["stage"]): row for row in trajectory["stages"]}
    if any(name not in rows_by_stage for name in stage_names):
        raise ValueError(f"trajectory {trajectory['record_id']} lacks stage bundle {stage}")
    if len(stage_names) == 1:
        visible = rows_by_stage[stage_names[0]]["visible_observation"]
        record_payload = dict(visible["record"])
        provenance = dict(visible["provenance"])
    else:
        base = rows_by_stage["base"]["visible_observation"]
        record_payload = dict(base["record"])
        evidence = list(record_payload.get("evidence", []))
        for name in stage_names:
            stage_record = rows_by_stage[name]["visible_observation"]["record"]
            for item in stage_record.get("evidence", []):
                if item not in evidence:
                    evidence.append(item)
        record_payload["evidence"] = evidence
        record_payload["metadata"] = {
            **record_payload.get("metadata", {}),
            "tool_stage_bundle": list(stage_names),
        }
        provenance = {
            "source_type": "cumulative_controlled_tool_replay",
            "hidden_outcome_exposed": False,
            "stage_bundle": list(stage_names),
        }
    if cost_aware:
        record_payload["features"] = {
            **record_payload.get("features", {}),
            "cost": float(record_payload.get("features", {}).get("cost", 0.0))
            + _acquired_cost(trajectory, stage),
        }
    record_payload = {
        **record_payload,
        "label": 0,
        "utility": 0.0,
    }
    record = ActionRecord.from_json(record_payload)
    return EvidenceView(
        view_id=(
            f"sequential-v2-cost-aware::{stage}"
            if cost_aware
            else f"sequential-v2::{stage}"
        ),
        record=record,
        intervention=f"sequential_tool_stage_bundle::{stage}",
        provenance=provenance,
    )


def _acquired_cost(trajectory, stage: str) -> float:
    stage_names = tuple(value for value in stage.split("+") if value and value != "base")
    rows_by_stage = {str(row["stage"]): row for row in trajectory["stages"]}
    return sum(float(rows_by_stage[name]["tool_cost"]) for name in stage_names)


def _reservation_values(trajectories):
    values = defaultdict(set)
    for row in trajectories:
        values[row["benchmark"]].add(
            float(row["hidden_outcome_for_evaluation_only"]["reservation_value"])
        )
    if any(len(task_values) != 1 for task_values in values.values()):
        raise ValueError("reservation value must be task-constant and train-derived")
    return {task: next(iter(task_values)) for task, task_values in sorted(values.items())}


def _summary(rows, models):
    labels = [row["worthy"] for row in rows]
    ensemble = [row["mean_p_positive_gain"] for row in rows]
    errors = [
        abs(row["mean_expected_gain"] - row["target_gain"])
        for row in rows
    ]
    output = {
        "ensemble": {
            **binary_metrics(labels, ensemble, threshold=0.5),
            "expected_gain_mae": _mean(errors),
            "probability_variance_error_correlation": _correlation(
                [row["probability_variance"] for row in rows], errors
            ),
            "expected_gain_variance_error_correlation": _correlation(
                [row["expected_gain_variance"] for row in rows], errors
            ),
        },
        "by_model": {},
        "by_task": {},
    }
    for model in models:
        probabilities = [row["models"][model]["p_positive_gain"] for row in rows]
        gains = [row["models"][model]["expected_gain"] for row in rows]
        output["by_model"][model] = {
            **binary_metrics(labels, probabilities, threshold=0.5),
            "expected_gain_mae": _mean(
                abs(prediction - row["target_gain"])
                for prediction, row in zip(gains, rows)
            ),
        }
    for task in sorted({row["benchmark"] for row in rows}):
        selected = [row for row in rows if row["benchmark"] == task]
        output["by_task"][task] = binary_metrics(
            [row["worthy"] for row in selected],
            [row["mean_p_positive_gain"] for row in selected],
            threshold=0.5,
        )
    return output


def _correlation(left, right):
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = (
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _variance(values):
    rows = list(values)
    center = _mean(rows)
    return _mean((value - center) ** 2 for value in rows)


if __name__ == "__main__":
    main()
