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
TOOL_EXECUTION_COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    args = parser.parse_args()
    result = run_single_task_voe(
        args.fold_dir,
        args.source,
        task=TASK,
        environment_class=ExternalH17PairwiseEvidenceEnvironment,
        experiment_name="h17_verify_before_execute_development",
        tool_decision_mode="direct_high_fidelity_gain",
        evidence_value_target_mode="cross_fitted_policy_conditioned",
        evidence_value_model_family="regularized_linear",
        base_sources=BASE_SOURCES,
        tool_sources=TOOL_SOURCES,
        base_execution_coverages=(0.0,),
        tool_execution_coverages=TOOL_EXECUTION_COVERAGES,
        seed=args.seed,
        crossfit_folds=args.crossfit_folds,
    )
    result["development_only"] = True
    result["consumed_folds_reused_for_contract_diagnosis"] = True
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "RESULTS.md").write_text(_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
