from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .historical import MAIN_MATERIAL_TASKS, grouped_four_way_split, load_historical_tasks
from .scientific_gain import BudgetReservationValues, EmpiricalUtilityNormalizer, GainConfig, gain_targets
from .sequential_tool_benchmark import SequentialToolEnvironment, audit_visible_payload


def export_sequential_benchmark_trajectories(
    raw_data_dir: str | Path,
    output_path: str | Path,
    *,
    seed: int = 7,
    per_task: int = 300,
    per_task_counts: dict[str, int] | None = None,
    budget_fraction: float = 0.10,
    exclude_trajectory_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    requested_counts = {task: int(per_task) for task in MAIN_MATERIAL_TASKS}
    if per_task_counts:
        unknown = set(per_task_counts) - set(MAIN_MATERIAL_TASKS)
        if unknown:
            raise ValueError(f"unknown task count overrides: {sorted(unknown)}")
        for task, count in per_task_counts.items():
            if int(count) < 1:
                raise ValueError("per-task trajectory counts must be positive")
            requested_counts[task] = int(count)
    records_by_task = load_historical_tasks(raw_data_dir, MAIN_MATERIAL_TASKS)
    excluded_fingerprints = _load_excluded_fingerprints(exclude_trajectory_paths)
    splits_by_task = {
        task: grouped_four_way_split(records, seed=seed)
        for task, records in records_by_task.items()
    }
    train_records = [
        record
        for task in MAIN_MATERIAL_TASKS
        for record in splits_by_task[task]["train"]
    ]
    all_records = [
        record
        for task in MAIN_MATERIAL_TASKS
        for split in ("train", "feedback", "acceptance", "test")
        for record in splits_by_task[task][split]
    ]
    normalizer = EmpiricalUtilityNormalizer.fit(train_records)
    gain_config = GainConfig()
    reservations = BudgetReservationValues.fit(
        train_records,
        normalizer,
        gain_config,
        budget_fraction=budget_fraction,
        outside_option=0.0,
    )
    targets_by_id = {
        target.record_id: target
        for target in gain_targets(all_records, normalizer, gain_config, reservations)
    }
    environment = SequentialToolEnvironment.fit(raw_data_dir, train_records)
    selected = []
    excluded_counts = {task: 0 for task in MAIN_MATERIAL_TASKS}
    duplicate_counts = {task: 0 for task in MAIN_MATERIAL_TASKS}
    for task in MAIN_MATERIAL_TASKS:
        task_records = []
        seen_task_fingerprints = set()
        for record in all_records:
            if record.benchmark != task:
                continue
            fingerprint = scientific_record_fingerprint(
                record.benchmark,
                record.visible_context,
                record.candidate_action,
            )
            if fingerprint in excluded_fingerprints:
                excluded_counts[task] += 1
                continue
            if fingerprint in seen_task_fingerprints:
                duplicate_counts[task] += 1
                continue
            seen_task_fingerprints.add(fingerprint)
            task_records.append(record)
        task_records.sort(key=lambda record: _stable_rank(seed, record.record_id))
        requested = requested_counts[task]
        if len(task_records) < requested:
            raise ValueError(
                f"task {task} has only {len(task_records)} non-excluded records; "
                f"requested {requested}"
            )
        selected.extend(task_records[:requested])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {task: 0 for task in MAIN_MATERIAL_TASKS}
    split_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            stages = []
            cumulative_cost = 0.0
            for step, stage in enumerate(("base", "screening", "verification")):
                observation = environment.observe(
                    record,
                    stage=stage,
                    evidence_replica=step,
                    run_seed=seed + 5000,
                )
                visible = observation.visible_payload()
                audit_visible_payload(visible)
                cumulative_cost += observation.tool_cost
                stages.append(
                    {
                        "step": step,
                        "stage": stage,
                        "tool_cost": observation.tool_cost,
                        "cumulative_tool_cost": cumulative_cost,
                        "visible_observation": visible,
                    }
                )
            target = targets_by_id[record.record_id]
            trajectory = {
                "trajectory_id": f"sd-worth-v2::{record.benchmark}::{record.record_id}",
                "benchmark_version": "sequential_tool_observation_v2",
                "benchmark": record.benchmark,
                "record_id": record.record_id,
                "split": record.split,
                "source": "historical_benchmark_controlled_replay",
                "online": False,
                "matbot_runtime": False,
                "semi_synthetic_observation_channel": True,
                "proposed_action": {
                    "action_type": record.action_type,
                    "description": record.candidate_action,
                },
                "stages": stages,
                "hidden_outcome_for_evaluation_only": {
                    "hidden_from_visible_stages": True,
                    "success_label": record.label,
                    "raw_utility": record.utility,
                    "normalized_utility": target.normalized_utility,
                    "base_gain": target.base_gain,
                    "reservation_value": target.reservation_value,
                    "budget_conditioned_gain": target.gain,
                    "worthy": target.worthy,
                    "latent_task_margin": environment.latent_margin(record),
                },
            }
            line = json.dumps(trajectory, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
            counts[record.benchmark] += 1
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
    manifest = {
        "benchmark_version": "sequential_tool_observation_v2",
        "path": str(output_path),
        "sha256": digest.hexdigest(),
        "seed": seed,
        "per_task": per_task,
        "requested_task_counts": requested_counts,
        "excluded_scientific_fingerprints": len(excluded_fingerprints),
        "excluded_source_records_by_task": excluded_counts,
        "duplicate_source_records_by_task": duplicate_counts,
        "scientific_fingerprint_overlap": 0,
        "total_trajectories": len(selected),
        "task_counts": counts,
        "split_counts": split_counts,
        "stages_per_trajectory": 3,
        "online": False,
        "matbot_runtime": False,
        "semi_synthetic_observation_channel": True,
        "hidden_outcomes_separated": True,
        "route_in_harness_output": False,
        "note": "Benchmark trajectories contain observations and outcomes, not evaluated H5.2 model outputs.",
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def scientific_record_fingerprint(
    benchmark: str,
    visible_context: str,
    candidate_action: str,
) -> str:
    payload = "\x1f".join(
        (
            str(benchmark).strip(),
            " ".join(str(visible_context).split()),
            " ".join(str(candidate_action).split()),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_excluded_fingerprints(paths: tuple[str | Path, ...]) -> set[str]:
    output = set()
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trajectory = json.loads(line)
                stages = {
                    str(row["stage"]): row
                    for row in trajectory.get(
                        "post_acquisition_observed_stages",
                        trajectory.get("stages", ()),
                    )
                }
                record = stages["base"]["visible_observation"]["record"]
                output.add(
                    scientific_record_fingerprint(
                        str(trajectory["benchmark"]),
                        str(record["visible_context"]),
                        str(record["candidate_action"]),
                    )
                )
    return output


def _stable_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|{record_id}".encode()).hexdigest()
