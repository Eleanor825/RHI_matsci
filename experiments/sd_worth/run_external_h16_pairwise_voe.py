from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h16_pairwise import TASK
from harness_matsci.external_h16_pairwise_tools import (
    ExternalH16PairwiseEvidenceEnvironment,
)

from run_external_h15_unique_voe import (
    _markdown,
    run_single_task_voe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    args = parser.parse_args()
    result = run_single_task_voe(
        args.fold_dir,
        args.source,
        task=TASK,
        environment_class=ExternalH16PairwiseEvidenceEnvironment,
        experiment_name="h16_powered_pairwise_two_stage_voe",
        tool_decision_mode="direct_high_fidelity_gain",
        evidence_value_target_mode="cross_fitted_policy_conditioned",
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
