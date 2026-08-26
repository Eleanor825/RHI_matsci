from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h16_pairwise import (
    build_external_h16_pairwise_fold,
    write_external_h16_pairwise_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    manifests = []
    for raw_fold in args.folds.split(","):
        fold = int(raw_fold)
        built = build_external_h16_pairwise_fold(args.source, fold=fold)
        manifest = write_external_h16_pairwise_fold(built, args.output_root)
        manifests.append(manifest)
        print(json.dumps({"fold": fold, "counts": manifest["counts"]}, sort_keys=True))
    root = Path(args.output_root)
    (root / "all_manifests.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
