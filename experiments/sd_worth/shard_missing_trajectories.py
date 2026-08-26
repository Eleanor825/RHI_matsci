from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--view-id", default="sequential-v2::base")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("shards must be positive")
    rows = [
        json.loads(line)
        for line in Path(args.trajectories).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    cached = set(payload.get("predictions", {}))
    missing = [
        row
        for row in rows
        if row["split"] == args.split
        and f"{row['record_id']}::{args.view_id}" not in cached
    ]
    buckets = [[] for _ in range(args.shards)]
    for row in missing:
        digest = hashlib.sha256(str(row["record_id"]).encode()).hexdigest()
        buckets[int(digest[:16], 16) % args.shards].append(row)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = []
    for index, bucket in enumerate(buckets):
        path = output / f"shard_{index:02d}.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) for row in bucket
            )
            + ("\n" if bucket else ""),
            encoding="utf-8",
        )
        counts.append(len(bucket))
    print(
        json.dumps(
            {
                "split": args.split,
                "cached_count": len(cached),
                "missing_count": len(missing),
                "shard_counts": counts,
                "output_dir": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
