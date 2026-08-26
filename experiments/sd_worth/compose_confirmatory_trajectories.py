from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--exclude-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("acceptance", "full"), required=True)
    args = parser.parse_args()
    expanded = _load(Path(args.input))
    excluded = {
        row["record_id"] for row in _load(Path(args.exclude_evaluation))
    }
    allowed = {"train", "feedback", "acceptance"}
    if args.phase == "full":
        allowed.add("test")
    selected = []
    for row in expanded:
        split = row["split"]
        if split not in allowed:
            continue
        if split in {"acceptance", "test"} and row["record_id"] in excluded:
            continue
        selected.append(row)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    manifest = {
        "phase": args.phase,
        "input": args.input,
        "exclude_evaluation": args.exclude_evaluation,
        "record_count": len(selected),
        "split_counts": {
            split: sum(row["split"] == split for row in selected)
            for split in sorted(allowed)
        },
        "evaluation_overlap_with_excluded": sum(
            row["split"] in {"acceptance", "test"} and row["record_id"] in excluded
            for row in selected
        ),
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
