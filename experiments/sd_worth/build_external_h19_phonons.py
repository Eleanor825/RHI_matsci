from __future__ import annotations

import argparse
import json

from harness_matsci.external_h19_phonon_pairwise import (
    build_external_h19_phonon_fold,
    write_external_h19_phonon_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--pair-multiplier", type=float, default=5.0)
    args = parser.parse_args()
    fold = build_external_h19_phonon_fold(
        args.snapshot_root,
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        pair_multiplier=args.pair_multiplier,
    )
    manifest = write_external_h19_phonon_fold(fold, args.output_root)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
