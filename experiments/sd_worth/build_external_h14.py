from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h14 import build_external_h14_fold, write_external_h14_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    manifests = []
    for value in args.folds.split(","):
        if not value.strip():
            continue
        fold = int(value)
        built = build_external_h14_fold(args.snapshot_root, fold=fold)
        manifest = write_external_h14_fold(built, args.output_root)
        manifests.append(manifest)
        print(json.dumps({"fold": fold, "counts": manifest["counts"]}, sort_keys=True))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "all_manifests.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
