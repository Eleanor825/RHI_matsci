from __future__ import annotations

import argparse
import json
from pathlib import Path


def finalize_cache(
    *,
    trajectories: str | Path,
    inputs: list[str | Path],
    output: str | Path,
    model: str,
    prompt_version: str,
    reasoning_effort: str,
    view_id: str = "sequential-v2::base",
) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one cache input is required")
    expected = {
        f"{row['record_id']}::{view_id}"
        for row in _load_jsonl(Path(trajectories))
    }
    merged = {}
    duplicates = 0
    extras = set()
    for path in inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        contract = (
            payload.get("model"),
            payload.get("prompt_version"),
            payload.get("reasoning_effort"),
        )
        if contract != (model, prompt_version, reasoning_effort):
            raise ValueError(f"cache contract mismatch: {path}")
        for key, value in payload.get("predictions", {}).items():
            if key not in expected:
                extras.add(key)
                continue
            if key in merged:
                duplicates += 1
                if merged[key] != value:
                    raise ValueError(f"conflicting prediction for {key}")
            merged[key] = value
    missing = expected - set(merged)
    if missing:
        raise ValueError(
            f"final cache is missing {len(missing)} predictions: {sorted(missing)[:3]}"
        )
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "reasoning_effort": reasoning_effort,
        "predictions": dict(sorted(merged.items())),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "model": model,
        "prompt_version": prompt_version,
        "reasoning_effort": reasoning_effort,
        "expected_count": len(expected),
        "prediction_count": len(merged),
        "identical_duplicate_count": duplicates,
        "pruned_extra_count": len(extras),
        "output": str(destination),
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--view-id", default="sequential-v2::base")
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_cache(
                trajectories=args.trajectories,
                inputs=args.inputs,
                output=args.output,
                model=args.model,
                prompt_version=args.prompt_version,
                reasoning_effort=args.reasoning_effort,
                view_id=args.view_id,
            ),
            sort_keys=True,
        )
    )


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
