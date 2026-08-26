from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness_matsci.historical import (
    MAIN_MATERIAL_TASKS,
    grouped_four_way_split,
    load_historical_tasks,
)
from harness_matsci.sequential_tool_benchmark import (
    SequentialToolEnvironment,
    audit_visible_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    manifest = build_crossfit_trajectories(
        args.raw_data_dir,
        args.input,
        args.output,
        seed=args.seed,
        folds=args.folds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_crossfit_trajectories(
    raw_data_dir,
    input_path,
    output_path,
    *,
    seed: int,
    folds: int,
):
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    trajectories = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_train_ids = {
        row["record_id"] for row in trajectories if row["split"] == "train"
    }
    records_by_task = load_historical_tasks(raw_data_dir, MAIN_MATERIAL_TASKS)
    train_records = [
        record
        for task in MAIN_MATERIAL_TASKS
        for record in grouped_four_way_split(records_by_task[task], seed=seed)["train"]
    ]
    record_by_id = {record.record_id: record for record in train_records}
    missing = sorted(selected_train_ids - set(record_by_id))
    if missing:
        raise ValueError(f"selected train trajectories missing raw records: {missing[:3]}")
    fold_by_group = {
        _group_key(record): _fold(seed, _group_key(record), folds)
        for record in train_records
    }
    replacement = {}
    audits = []
    for fold in range(folds):
        held_groups = {
            group for group, assigned in fold_by_group.items() if assigned == fold
        }
        fit_records = [
            record for record in train_records if _group_key(record) not in held_groups
        ]
        held_records = [
            record_by_id[record_id]
            for record_id in sorted(selected_train_ids)
            if fold_by_group[_group_key(record_by_id[record_id])] == fold
        ]
        if not fit_records or not held_records:
            raise ValueError(f"crossfit fold {fold} has empty fit or held records")
        environment = SequentialToolEnvironment.fit(raw_data_dir, fit_records)
        for record in held_records:
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
            replacement[record.record_id] = stages
        audits.append(
            {
                "fold": fold,
                "fit_record_count": len(fit_records),
                "held_selected_count": len(held_records),
                "fit_group_count": len({_group_key(record) for record in fit_records}),
                "held_group_count": len(held_groups),
                "group_overlap": sorted(
                    {_group_key(record) for record in fit_records} & held_groups
                ),
            }
        )
    if set(replacement) != selected_train_ids:
        raise RuntimeError("cross-fitting did not rebuild every selected train trajectory")
    output = []
    digest = hashlib.sha256()
    for trajectory in trajectories:
        rebuilt = dict(trajectory)
        if trajectory["split"] == "train":
            rebuilt["stages"] = replacement[trajectory["record_id"]]
            rebuilt["train_tool_observation_mode"] = "group_excluded_5_fold_crossfit"
        line = json.dumps(rebuilt, ensure_ascii=False, sort_keys=True)
        output.append(line)
        digest.update((line + "\n").encode())
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    manifest = {
        "benchmark_version": "sequential_tool_observation_v3_crossfit_train",
        "source": str(input_path),
        "output": str(destination),
        "sha256": digest.hexdigest(),
        "seed": seed,
        "crossfit_folds": folds,
        "trajectory_count": len(trajectories),
        "crossfit_train_count": len(selected_train_ids),
        "nontrain_unchanged_count": len(trajectories) - len(selected_train_ids),
        "group_exclusion_audits": audits,
        "all_group_overlaps_empty": all(not row["group_overlap"] for row in audits),
        "hidden_outcomes_separated": True,
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _group_key(record) -> str:
    return f"{record.benchmark}::{record.metadata.get('group_id', record.record_id)}"


def _fold(seed: int, group: str, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|crossfit|{group}".encode()).hexdigest()
    return int(digest[:16], 16) % folds


if __name__ == "__main__":
    main()
