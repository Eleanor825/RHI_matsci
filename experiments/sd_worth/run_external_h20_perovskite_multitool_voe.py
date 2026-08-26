from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h20_perovskite_pairwise import TASK
from harness_matsci.external_h20_perovskite_tools import (
    ExternalH20PerovskiteEvidenceEnvironment,
)

from run_external_h17_multitool_voe import _markdown, run_multitool_voe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    args = parser.parse_args()
    result = run_multitool_voe(
        args.fold_dir,
        args.snapshot_root,
        task=TASK,
        environment_class=ExternalH20PerovskiteEvidenceEnvironment,
        experiment_name="h20_perovskite_multitool_external_validation",
        development_only=False,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
        policy_selection_mode="absolute_threshold",
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
