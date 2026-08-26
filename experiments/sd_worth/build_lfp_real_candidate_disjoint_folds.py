from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.lfp_real_splits import (
    audit_candidate_disjoint_fold,
    build_candidate_disjoint_lfp_folds,
)
from harness_matsci.schema import ActionRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    records = [
        ActionRecord.from_json(json.loads(line))
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = []
    for fold in build_candidate_disjoint_lfp_folds(records, folds=args.folds):
        fold_dir = destination / f"fold_{fold.fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "feedback", "acceptance", "test"):
            rows = getattr(fold, split)
            (fold_dir / f"{split}.jsonl").write_text(
                "\n".join(
                    json.dumps(row.to_json(), ensure_ascii=False, sort_keys=True)
                    for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
        manifest = {
            **fold.to_manifest(),
            "audit": audit_candidate_disjoint_fold(fold),
        }
        (fold_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)
    (destination / "manifest.json").write_text(
        json.dumps({"folds": manifests}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"folds": manifests}, sort_keys=True))


if __name__ == "__main__":
    main()
