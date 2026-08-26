from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.external_h19_phonon_pairwise import TASK
from harness_matsci.external_h19_phonon_tools import ExternalH19PhononEvidenceEnvironment

from run_external_h15_unique_voe import _markdown, run_single_task_voe


BASE_SOURCES = ("agent_prior", "composition", "structure")
TOOL_SOURCES = (*BASE_SOURCES, "high_fidelity")
TOOL_EXECUTION_COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--include-record-predictions", action="store_true")
    parser.add_argument(
        "--policy-selection-mode",
        choices=("absolute_threshold", "rank_budget"),
        default="absolute_threshold",
    )
    args = parser.parse_args()
    result = run_single_task_voe(
        args.fold_dir,
        args.snapshot_root,
        task=TASK,
        environment_class=ExternalH19PhononEvidenceEnvironment,
        experiment_name="h19_phonons_frozen_external_validation",
        tool_decision_mode="direct_high_fidelity_gain",
        evidence_value_target_mode="cross_fitted_policy_conditioned",
        evidence_value_model_family="regularized_linear",
        base_sources=BASE_SOURCES,
        tool_sources=TOOL_SOURCES,
        base_execution_coverages=(0.0,),
        tool_execution_coverages=TOOL_EXECUTION_COVERAGES,
        include_record_predictions=args.include_record_predictions,
        development_only=False,
        policy_selection_mode=args.policy_selection_mode,
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
