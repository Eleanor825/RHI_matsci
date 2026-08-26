from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h22_extreme_gap import TASK
from harness_matsci.external_h22_extreme_gap_tools import (
    ExternalH22ExtremeGapEnvironment,
)

from run_external_h15_unique_voe import _markdown, run_single_task_voe


BASE_SOURCES = ("coarse_composition", "full_composition")
TOOL_SOURCES = (*BASE_SOURCES, "high_fidelity")
TOOL_EXECUTION_COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument(
        "--policy-selection-mode",
        choices=("absolute_threshold", "rank_budget"),
        default="absolute_threshold",
    )
    args = parser.parse_args()
    result = run_single_task_voe(
        args.fold_dir,
        args.source,
        task=TASK,
        environment_class=ExternalH22ExtremeGapEnvironment,
        experiment_name="h22_extreme_gap_verify_before_execute",
        tool_decision_mode="direct_high_fidelity_gain",
        evidence_value_target_mode="cross_fitted_policy_conditioned",
        evidence_value_model_family="regularized_linear",
        base_sources=BASE_SOURCES,
        tool_sources=TOOL_SOURCES,
        base_execution_coverages=(0.0,),
        tool_execution_coverages=TOOL_EXECUTION_COVERAGES,
        development_only=not args.confirmatory,
        include_tail_methods=False,
        alpha=args.alpha,
        delta=0.05,
        policy_selection_mode=args.policy_selection_mode,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (root / "RESULTS.md").write_text(_markdown(result))


if __name__ == "__main__":
    main()
