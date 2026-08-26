from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h22_extreme_gap import (
    build_external_h22_fold,
    write_external_h22_fold,
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
        manifest = write_external_h22_fold(
            build_external_h22_fold(args.source, fold=fold),
            args.output_root,
        )
        manifests.append(manifest)
        print(json.dumps({"fold": fold, "counts": manifest["counts"]}, sort_keys=True))
    Path(args.output_root, "all_manifests.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
