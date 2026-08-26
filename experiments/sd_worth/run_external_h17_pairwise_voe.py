from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h17_pairwise import TASK
from harness_matsci.external_h17_pairwise_tools import (
    ExternalH17PairwiseEvidenceEnvironment,
)

from run_external_h15_unique_voe import _markdown, run_single_task_voe


BASE_SOURCES = ("agent_prior", "composition", "structure")
TOOL_SOURCES = (*BASE_SOURCES, "high_fidelity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--verify-before-execute", action="store_true")
    parser.add_argument("--decision-instability", action="store_true")
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
        environment_class=ExternalH17PairwiseEvidenceEnvironment,
        experiment_name="h17_mixed_regime_pairwise_two_stage_voe",
        tool_decision_mode="direct_high_fidelity_gain",
        evidence_value_target_mode="cross_fitted_policy_conditioned",
        evidence_value_model_family="regularized_linear",
        base_sources=BASE_SOURCES,
        tool_sources=TOOL_SOURCES,
        base_execution_coverages=(0.0,) if args.verify_before_execute else None,
        policy_selection_mode=args.policy_selection_mode,
        use_decision_instability=args.decision_instability,
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
