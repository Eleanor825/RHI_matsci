from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_cache_files(inputs: list[str | Path], output: str | Path) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one cache input is required")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in inputs]
    contract = {
        key: payloads[0].get(key)
        for key in ("model", "prompt_version", "reasoning_effort")
    }
    if any(
        any(payload.get(key) != value for key, value in contract.items())
        for payload in payloads[1:]
    ):
        raise ValueError("cache contracts do not match")
    predictions: dict[str, object] = {}
    duplicate_count = 0
    for payload in payloads:
        for key, value in payload.get("predictions", {}).items():
            if key in predictions:
                duplicate_count += 1
                if predictions[key] != value:
                    raise ValueError(f"conflicting prediction for cache key {key}")
            predictions[key] = value
    merged = {
        **contract,
        "predictions": dict(sorted(predictions.items())),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        **contract,
        "input_count": len(inputs),
        "prediction_count": len(predictions),
        "identical_duplicate_count": duplicate_count,
        "output": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge_cache_files(args.inputs, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
