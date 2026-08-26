from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness_matsci.historical import MAIN_MATERIAL_TASKS, load_historical_tasks
from harness_matsci.sequential_tool_benchmark import (
    RealScientificToolEnvironment,
    TrainOnlyStructuredSurrogateToolEnvironment,
    TrainOnlySurrogateToolEnvironment,
    audit_visible_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7729)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument(
        "--environment",
        choices=("structured", "real", "text"),
        default="structured",
    )
    args = parser.parse_args()
    report = build_adaptive_evidence_trajectories(
        raw_data_dir=args.raw_data_dir,
        input_path=args.input,
        output_path=args.output,
        seed=args.seed,
        folds=args.folds,
        replicas=args.replicas,
        environment_name=args.environment,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_adaptive_evidence_trajectories(
    *,
    raw_data_dir,
    input_path,
    output_path,
    seed,
    folds,
    replicas,
    environment_name="structured",
):
    if folds < 2 or replicas < 2:
        raise ValueError("at least two folds and two evidence replicas are required")
    trajectories = _load_jsonl(Path(input_path))
    records_by_task = load_historical_tasks(raw_data_dir, MAIN_MATERIAL_TASKS)
    record_by_id = {
        record.record_id: record
        for task in MAIN_MATERIAL_TASKS
        for record in records_by_task[task]
    }
    missing = sorted({row["record_id"] for row in trajectories} - set(record_by_id))
    if missing:
        raise ValueError(f"input trajectories lack raw records: {missing[:3]}")
    train_records = tuple(
        record_by_id[row["record_id"]]
        for row in trajectories
        if row["split"] == "train"
    )
    if not train_records:
        raise ValueError("input trajectories contain no training records")
    fold_by_group = {
        _group_key(record): _fold(seed, _group_key(record), folds)
        for record in train_records
    }
    train_prior_by_id = {}
    fold_audits = []
    for fold in range(folds):
        held_groups = {
            group for group, assigned in fold_by_group.items() if assigned == fold
        }
        fit_records = tuple(
            record for record in train_records if _group_key(record) not in held_groups
        )
        held_records = tuple(
            record for record in train_records if _group_key(record) in held_groups
        )
        if not fit_records or not held_records:
            raise ValueError(f"adaptive evidence fold {fold} is empty")
        environment = _environment_class(environment_name).fit(
            raw_data_dir,
            fit_records,
            environment_seed=seed + fold,
        )
        for record in held_records:
            train_prior_by_id[record.record_id] = _prior_views(
                environment,
                record,
                replicas=replicas,
                run_seed=seed + 1000 + fold,
            )
        fold_audits.append(
            {
                "fold": fold,
                "fit_record_count": len(fit_records),
                "held_record_count": len(held_records),
                "group_overlap": sorted(
                    {_group_key(record) for record in fit_records} & held_groups
                ),
                "training_hash": environment.training_hash,
            }
        )
    deployment_environment = _environment_class(environment_name).fit(
        raw_data_dir,
        train_records,
        environment_seed=seed + folds,
    )
    output_lines = []
    digest = hashlib.sha256()
    split_counts = {}
    for trajectory in trajectories:
        split = str(trajectory["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        record = record_by_id[trajectory["record_id"]]
        prior_views = (
            train_prior_by_id[record.record_id]
            if split == "train"
            else _prior_views(
                deployment_environment,
                record,
                replicas=replicas,
                run_seed=seed + 2000,
            )
        )
        rebuilt = {
            **trajectory,
            "schema_version": "adaptive-evidence-trajectory-v1",
            "pre_acquisition_prior_views": prior_views,
            "post_acquisition_observed_stages": trajectory["stages"],
            "evidence_contract": {
                "pre_acquisition": "train-only surrogate replicas",
                "post_acquisition": "controlled noisy task-native measurement",
                "hidden_outcome_in_pre_acquisition": False,
                "tool_cost_charged_after_acquisition": True,
                "replicas_per_tool": replicas,
            },
        }
        line = json.dumps(rebuilt, ensure_ascii=False, sort_keys=True)
        output_lines.append(line)
        digest.update((line + "\n").encode("utf-8"))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    report = {
        "schema_version": "adaptive-evidence-trajectory-manifest-v1",
        "source": str(input_path),
        "output": str(destination),
        "sha256": digest.hexdigest(),
        "trajectory_count": len(trajectories),
        "split_counts": dict(sorted(split_counts.items())),
        "seed": seed,
        "folds": folds,
        "replicas": replicas,
        "environment": environment_name,
        "all_train_group_overlaps_empty": all(
            not row["group_overlap"] for row in fold_audits
        ),
        "fold_audits": fold_audits,
        "deployment_training_hash": deployment_environment.training_hash,
        "claim_boundary": {
            "prior_views_use_held_out_outcome": False,
            "observed_tool_views_are_available_only_after_acquisition": True,
            "matbot_runtime": False,
        },
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _prior_views(environment, record, *, replicas, run_seed):
    output = []
    for stage in ("screening", "verification"):
        for replica in range(replicas):
            observation = environment.observe(
                record,
                stage=stage,
                evidence_replica=replica,
                run_seed=run_seed,
            )
            visible = observation.visible_payload()
            audit_visible_payload(visible)
            output.append(
                {
                    "stage": stage,
                    "evidence_replica": replica,
                    "tool_cost": observation.tool_cost,
                    "visible_observation": visible,
                }
            )
    return output


def _environment_class(name):
    return {
        "structured": TrainOnlyStructuredSurrogateToolEnvironment,
        "real": RealScientificToolEnvironment,
        "text": TrainOnlySurrogateToolEnvironment,
    }[name]


def _group_key(record):
    return f"{record.benchmark}::{record.metadata.get('group_id', record.record_id)}"


def _fold(seed, group, folds):
    digest = hashlib.sha256(f"{seed}|adaptive-evidence|{group}".encode()).hexdigest()
    return int(digest[:16], 16) % folds


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
