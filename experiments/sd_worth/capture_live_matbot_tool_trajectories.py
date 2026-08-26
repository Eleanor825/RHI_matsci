from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_matsci.factorized_gain_harness import FactorizedGainPrediction
from harness_matsci.matbot_strict_gain_runtime import (
    append_runtime_event,
    strict_gain_post_outcome_event,
    strict_gain_pre_action_event,
)
from harness_matsci.matbot_trajectory import assemble_matbot_trajectories, trajectory_summary
from harness_matsci.real_scientific_tools import RealScientificToolSuite


TASKS = ("matbench_pairwise", "discover_unique", "extreme_properties")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-data", required=True)
    parser.add_argument("--evaluation-sequential-data")
    parser.add_argument("--raw-data-dir", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="feedback")
    parser.add_argument("--n-per-task", type=int, default=300)
    parser.add_argument("--task-count", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    requested_counts = {task: args.n_per_task for task in TASKS}
    for item in args.task_count:
        task, value = item.split("=", 1)
        if task not in TASKS:
            raise ValueError(f"unknown task count override {task}")
        requested_counts[task] = int(value)

    adaptive_rows = _read_jsonl(Path(args.adaptive_data))
    raw_directories = [Path(value) for value in args.raw_data_dir]
    raw_by_id = _raw_lookup(raw_directories)
    training_rows = []
    evaluation_rows = []
    task_counts: Counter[str] = Counter()
    for adaptive in adaptive_rows:
        visible = adaptive["post_acquisition_observed_stages"][0]["visible_observation"]["record"]
        record_id = str(adaptive["record_id"])
        raw = raw_by_id.get(record_id)
        if raw is None:
            continue
        joined = {
            "benchmark": adaptive["benchmark"],
            "record_id": record_id,
            "visible_context": visible["visible_context"],
            "candidate_action": visible["candidate_action"],
            "evidence": visible["evidence"],
            "features": visible["features"],
            "hidden_outcome": adaptive["hidden_outcome_for_evaluation_only"],
            "raw_record": raw,
            "split": visible["split"],
        }
        if visible["split"] == "train":
            training_rows.append(joined)
        elif not args.evaluation_sequential_data and (
            visible["split"] == args.split
            and task_counts[adaptive["benchmark"]] < requested_counts[adaptive["benchmark"]]
        ):
            evaluation_rows.append(joined)
            task_counts[adaptive["benchmark"]] += 1

    if args.evaluation_sequential_data:
        evaluation_rows = []
        task_counts.clear()
        for trajectory in _read_jsonl(Path(args.evaluation_sequential_data)):
            if trajectory["split"] != args.split:
                continue
            benchmark = trajectory["benchmark"]
            if task_counts[benchmark] >= requested_counts[benchmark]:
                continue
            visible = next(
                stage["visible_observation"]["record"]
                for stage in trajectory["stages"]
                if stage["stage"] == "base"
            )
            record_id = str(trajectory["record_id"])
            raw = raw_by_id.get(record_id)
            if raw is None:
                continue
            evaluation_rows.append(
                {
                    "benchmark": benchmark,
                    "record_id": record_id,
                    "visible_context": visible["visible_context"],
                    "candidate_action": visible["candidate_action"],
                    "evidence": visible["evidence"],
                    "features": visible["features"],
                    "hidden_outcome": trajectory["hidden_outcome_for_evaluation_only"],
                    "raw_record": raw,
                    "split": trajectory["split"],
                }
            )
            task_counts[benchmark] += 1

    missing = [task for task in TASKS if task_counts[task] < requested_counts[task]]
    if missing:
        raise ValueError(f"insufficient rows for tasks: {missing}; counts={dict(task_counts)}")

    suite = RealScientificToolSuite(random_state=args.seed).fit(training_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    for path in (events_path, trajectories_path):
        if path.exists():
            path.unlink()

    events: list[dict[str, Any]] = []
    run_id = f"local-matbot-tools-{args.split}-{args.seed}"
    for index, row in enumerate(evaluation_rows):
        observation = suite.observe(row)
        trajectory_id = f"live-tool::{row['benchmark']}::{row['record_id']}"
        timestamp = datetime.now(timezone.utc).isoformat()
        features = row["features"]
        base_prediction = _prediction_from_visible(row)
        tool_event_id = f"{trajectory_id}::tool"
        tool_pre = strict_gain_pre_action_event(
            event_id=tool_event_id,
            run_id=run_id,
            trajectory_id=trajectory_id,
            step_id=0,
            timestamp=timestamp,
            system="matbot_local_scientific_tool_runtime",
            stage="base",
            action_id="acquire_scientific_tool_evidence",
            action_type="tool_call",
            visible_context=row["visible_context"],
            evidence=list(row["evidence"]),
            internal_signals=_internal_signals(features),
            external_signals=_external_signals(features),
            normalized_cost=0.025,
            reversibility=1.0,
            prediction=base_prediction,
            conformal_lower_gain=base_prediction.expected_net_gain
            - 1.645 * math.sqrt(base_prediction.total_gain_variance),
            capture_mode="live_runtime",
            agent_backend="deterministic_numeric_harness",
            action_text=f"Call {observation.tool_name} before deciding whether to execute.",
            rationale="Acquire task-native evidence using an actually executed scientific tool.",
        )
        tool_post = _tool_completion_event(
            event_id=f"{tool_event_id}::outcome",
            parent_event_id=tool_event_id,
            run_id=run_id,
            trajectory_id=trajectory_id,
            step_id=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            observation=observation.to_json(),
        )
        execution_prediction = _prediction_from_tool(row, observation.outputs)
        execute_event_id = f"{trajectory_id}::execute"
        execute_pre = strict_gain_pre_action_event(
            event_id=execute_event_id,
            run_id=run_id,
            trajectory_id=trajectory_id,
            step_id=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            system="matbot_local_scientific_tool_runtime",
            stage="real_tool_evidence",
            action_id="execute_candidate",
            action_type="choose_candidate",
            visible_context=row["visible_context"],
            evidence=list(row["evidence"]) + _observation_evidence(observation.outputs),
            internal_signals=_internal_signals(features),
            external_signals={
                **_external_signals(features),
                "real_tool_predictive_standard_deviation": observation.outputs.get(
                    "predictive_standard_deviation", 0.0
                ),
                "real_tool_positive_probability": observation.outputs.get(
                    "positive_probability", 0.5
                ),
            },
            normalized_cost=float(features.get("cost", 0.5)),
            reversibility=float(features.get("reversibility", 0.5)),
            prediction=execution_prediction,
            conformal_lower_gain=execution_prediction.expected_net_gain
            - 1.645 * math.sqrt(execution_prediction.total_gain_variance),
            capture_mode="live_runtime",
            agent_backend="deterministic_numeric_harness",
            tool_trace=[observation.to_json()],
            action_text=row["candidate_action"],
            rationale="Score the candidate numerically after observing real tool evidence.",
        )
        hidden = row["hidden_outcome"]
        success = int(hidden["success_label"])
        execute_post = strict_gain_post_outcome_event(
            event_id=f"{execute_event_id}::outcome",
            parent_event_id=execute_event_id,
            run_id=run_id,
            trajectory_id=trajectory_id,
            step_id=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            observed_success=success,
            normalized_utility=float(hidden["normalized_utility"]),
            realized_cost=float(features.get("cost", 0.5)),
            failure_harm=1.0,
            terminal=True,
            capture_mode="live_runtime",
        )
        for event in (tool_pre, tool_post, execute_pre, execute_post):
            append_runtime_event(events_path, event)
            events.append(event)

    trajectories = assemble_matbot_trajectories(events)
    with trajectories_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "live-matbot-scientific-tools-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adaptive_data": str(Path(args.adaptive_data).resolve()),
        "adaptive_data_sha256": _sha256(Path(args.adaptive_data)),
        "evaluation_sequential_data": (
            str(Path(args.evaluation_sequential_data).resolve())
            if args.evaluation_sequential_data
            else None
        ),
        "evaluation_sequential_data_sha256": (
            _sha256(Path(args.evaluation_sequential_data))
            if args.evaluation_sequential_data
            else None
        ),
        "raw_data_dirs": [str(path.resolve()) for path in raw_directories],
        "split": args.split,
        "seed": args.seed,
        "requested_counts": requested_counts,
        "counts": dict(sorted(task_counts.items())),
        "trajectory_summary": trajectory_summary(trajectories),
        "events_sha256": _sha256(events_path),
        "trajectories_sha256": _sha256(trajectories_path),
        "claim_boundary": (
            "Scientific tools were executed live over historical benchmark candidates; "
            "terminal outcomes are historical held-out outcomes, not new wet-lab measurements."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _prediction_from_visible(row: dict[str, Any]) -> FactorizedGainPrediction:
    features = row["features"]
    internal_components = (
        float(features.get("verbal_confidence", 0.5)),
        float(features.get("perturbation_stability", 0.5)),
        float(features.get("estimate_agreement", 0.5)),
    )
    external_components = (
        float(features.get("evidence_support", 0.5)),
        1.0 - float(features.get("evidence_conflict", 0.0)),
        1.0 - float(features.get("ood_score", 0.5)),
        float(features.get("source_reliability", 0.5)),
        1.0 - float(features.get("source_risk", 0.5)),
    )
    internal_mean = sum(internal_components) / len(internal_components)
    external_mean = sum(external_components) / len(external_components)
    probability = _clip01(0.5 * internal_mean + 0.5 * external_mean)
    utility = probability
    internal = _population_variance(internal_components)
    external = _population_variance(external_components)
    return _prediction(row, probability, utility, internal, external)


def _prediction_from_tool(
    row: dict[str, Any], outputs: dict[str, float]
) -> FactorizedGainPrediction:
    base = _prediction_from_visible(row)
    tool_probability = _clip01(outputs.get("positive_probability", 0.5))
    probability = _clip01(0.35 * base.success_probability + 0.65 * tool_probability)
    tool_utility = _clip01(outputs.get("estimated_utility", tool_probability))
    utility = _clip01(0.25 * base.conditional_utility_mean + 0.75 * tool_utility)
    external = base.probability_external_variance + float(
        outputs.get("predictive_standard_deviation", 0.1)
    ) ** 2
    internal = base.probability_internal_variance
    return _prediction(row, probability, utility, internal, external)


def _prediction(
    row: dict[str, Any], probability: float, utility: float, internal: float, external: float
) -> FactorizedGainPrediction:
    cost = float(row["features"].get("cost", 0.5))
    expected_gain = probability * utility - 0.15 * cost - 0.25 * (1.0 - probability)
    interaction = 0.1 * math.sqrt(max(0.0, internal * external))
    aleatoric = probability * (1.0 - probability) * utility**2
    total = internal + external + interaction + aleatoric
    worthiness = _clip01(0.5 + expected_gain / max(0.1, 4.0 * math.sqrt(total + 1e-9)))
    return FactorizedGainPrediction(
        record_id=row["record_id"],
        benchmark=row["benchmark"],
        action_worthiness=worthiness,
        expected_net_gain=expected_gain,
        success_probability=probability,
        conditional_utility_mean=utility,
        probability_internal_variance=internal,
        probability_external_variance=external,
        probability_interaction_variance=interaction,
        gain_internal_variance=internal,
        gain_external_variance=external,
        gain_interaction_variance=interaction,
        aleatoric_gain_variance=aleatoric,
        total_gain_variance=total,
    )


def _tool_completion_event(**kwargs) -> dict[str, Any]:
    observation = kwargs.pop("observation")
    return {
        "schema_version": "matxbot-uncertainty-v1",
        "event_type": "post_outcome",
        **kwargs,
        "capture_mode": "live_runtime",
        "outcome_status": "tool_completed",
        "observed_outcome": {
            "tool_success": 1,
            "tool_observation": observation,
        },
        "realized_cost": 0.025,
        "terminal": False,
    }


def _internal_signals(features: dict[str, Any]) -> dict[str, float]:
    return {
        "verbal_confidence": float(features.get("verbal_confidence", 0.5)),
        "perturbation_stability": float(features.get("perturbation_stability", 0.5)),
        "estimate_agreement": float(features.get("estimate_agreement", 0.5)),
    }


def _external_signals(features: dict[str, Any]) -> dict[str, float]:
    return {
        "evidence_support": float(features.get("evidence_support", 0.5)),
        "evidence_conflict": float(features.get("evidence_conflict", 0.0)),
        "ood_score": float(features.get("ood_score", 0.5)),
        "source_reliability": float(features.get("source_reliability", 0.5)),
        "source_risk": float(features.get("source_risk", 0.5)),
    }


def _observation_evidence(outputs: dict[str, float]) -> list[str]:
    preferred = (
        "positive_probability",
        "estimated_utility",
        "predictive_standard_deviation",
        "predicted_margin",
        "composition_structure_novelty",
        "six_descriptor_hit_fraction",
    )
    return [f"real_tool_{key}={outputs[key]:.6f}" for key in preferred if key in outputs]


def _raw_lookup(directories: list[Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for directory in directories:
        for filename in (
            "matbench_pairwise_actions.jsonl",
            "unique_materials_actions.jsonl",
            "extreme_properties_actions.jsonl",
        ):
            path = directory / filename
            if not path.exists():
                continue
            for row in _read_jsonl(path):
                benchmark = {
                    "materials_pairwise_preference": "matbench_pairwise",
                    "unique_materials_screening": "discover_unique",
                    "extreme_property_discovery": "extreme_properties",
                }[row["task"]]
                row = dict(row)
                row["benchmark"] = benchmark
                output[str(row["record_id"])] = row
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _population_variance(values) -> float:
    values = tuple(float(value) for value in values)
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
