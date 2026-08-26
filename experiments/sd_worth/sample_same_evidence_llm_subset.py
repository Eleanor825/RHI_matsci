from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--count", action="append", required=True)
    args = parser.parse_args()
    requested = {}
    for item in args.count:
        task, value = item.split("=", 1)
        requested[task] = int(value)
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["benchmark"], []).append(row)
    selected = []
    for benchmark, count in requested.items():
        candidates = sorted(
            grouped.get(benchmark, []),
            key=lambda row: hashlib.sha256(
                f"{args.seed}|{benchmark}|{row['record_id']}".encode()
            ).hexdigest(),
        )
        if len(candidates) < count:
            raise ValueError(f"requested {count} {benchmark}, found {len(candidates)}")
        selected.extend(candidates[:count])
    selected.sort(key=lambda row: (row["benchmark"], row["record_id"]))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter(row["benchmark"] for row in selected)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(json.dumps({"count": len(selected), "counts": counts, "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
