from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness_matsci.llm_tool_trajectory import (
    LLMToolPolicyClient,
    run_model_tool_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.trajectories))
    rows = [row for row in rows if row["split"] == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = LLMToolPolicyClient.from_env(
        model=args.model,
        cache_path=output / "cache" / f"{args.model}.json",
        reasoning_effort=args.reasoning_effort,
    )
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_model_tool_trajectory, row, client): row["record_id"]
            for row in rows
        }
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda row: row["source_record_id"])
    destination = output / "trajectories.jsonl"
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in completed),
        encoding="utf-8",
    )
    summary = {
        "model": args.model,
        "split": args.split,
        "trajectory_count": len(completed),
        "executed_count": sum(row["executed_candidate"] for row in completed),
        "mean_steps": sum(len(row["steps"]) for row in completed) / len(completed)
        if completed
        else 0.0,
        "mean_realized_gain": sum(
            row["hidden_outcome_for_evaluation_only"]["trajectory_strict_net_gain"]
            for row in completed
        )
        / len(completed)
        if completed
        else 0.0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
