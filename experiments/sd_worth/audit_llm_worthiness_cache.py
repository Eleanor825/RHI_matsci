from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--view-id", default="sequential-v2::base")
    args = parser.parse_args()
    trajectories = [
        json.loads(line)
        for line in Path(args.trajectories).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    expected = {
        f"{row['record_id']}::{args.view_id}" for row in trajectories
    }
    actual = set(payload.get("predictions", {}))
    report = {
        "model_match": payload.get("model") == args.model,
        "prompt_match": payload.get("prompt_version") == args.prompt_version,
        "reasoning_match": payload.get("reasoning_effort") == args.reasoning_effort,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_count": len(expected - actual),
        "extra_count": len(actual - expected),
        "missing_examples": sorted(expected - actual)[:5],
        "extra_examples": sorted(actual - expected)[:5],
    }
    report["complete"] = all(
        (
            report["model_match"],
            report["prompt_match"],
            report["reasoning_match"],
            report["missing_count"] == 0,
        )
    )
    print(json.dumps(report, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
