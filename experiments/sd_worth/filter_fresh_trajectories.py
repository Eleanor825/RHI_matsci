from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="acceptance,test")
    args = parser.parse_args()
    excluded = {
        row["record_id"] for row in _load(Path(args.exclude))
    }
    allowed = {value for value in args.splits.split(",") if value}
    rows = [
        row
        for row in _load(Path(args.input))
        if row["split"] in allowed and row["record_id"] not in excluded
    ]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    manifest = {
        "input": args.input,
        "exclude": args.exclude,
        "splits": sorted(allowed),
        "record_count": len(rows),
        "task_counts": {
            task: sum(row["benchmark"] == task for row in rows)
            for task in sorted({row["benchmark"] for row in rows})
        },
        "sha256": digest.hexdigest(),
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


def _load(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
