from __future__ import annotations

import argparse
import json

from harness_matsci.sequential_trajectory_export import export_sequential_benchmark_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-task", type=int, default=300)
    parser.add_argument(
        "--task-count",
        action="append",
        default=[],
        metavar="TASK=COUNT",
    )
    parser.add_argument("--exclude-trajectories", action="append", default=[])
    args = parser.parse_args()
    task_counts = {}
    for value in args.task_count:
        task, separator, count = value.partition("=")
        if not separator:
            raise ValueError("--task-count must use TASK=COUNT")
        task_counts[task] = int(count)
    manifest = export_sequential_benchmark_trajectories(
        args.raw_data_dir,
        args.output,
        seed=args.seed,
        per_task=args.per_task,
        per_task_counts=task_counts,
        exclude_trajectory_paths=tuple(args.exclude_trajectories),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
