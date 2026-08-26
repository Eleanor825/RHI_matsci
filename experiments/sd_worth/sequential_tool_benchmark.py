from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.sequential_tool_experiment import run_sequential_tool_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="1,7,13,21,42")
    parser.add_argument("--folds")
    parser.add_argument("--n-regime-folds", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--feedback-fraction", type=float, default=0.15)
    parser.add_argument("--acceptance-fraction", type=float, default=0.10)
    parser.add_argument(
        "--reservation-strategy",
        choices=("outcome_conditioned_base_gain", "potential_success_slot_value"),
        default="outcome_conditioned_base_gain",
    )
    parser.add_argument(
        "--environment-kind",
        choices=(
            "semi_synthetic",
            "train_only_surrogate",
            "train_only_structured_surrogate",
            "train_only_structured_direct",
            "real_scientific_tools",
        ),
        default="semi_synthetic",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    if args.folds:
        runs = [
            (1729 + int(value), int(value))
            for value in args.folds.split(",")
            if value.strip()
        ]
    else:
        runs = [
            (int(value), None)
            for value in args.seeds.split(",")
            if value.strip()
        ]
    for seed, fold in runs:
        result = run_sequential_tool_experiment(
            args.raw_data_dir,
            seed=seed,
            environment_kind=args.environment_kind,
            regime_fold=fold,
            n_regime_folds=args.n_regime_folds,
            reservation_strategy=args.reservation_strategy,
            train_fraction=args.train_fraction,
            feedback_fraction=args.feedback_fraction,
            acceptance_fraction=args.acceptance_fraction,
        )
        path = output_dir / (f"fold_{fold}.json" if fold is not None else f"seed_{seed}.json")
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(result)
        print(json.dumps(_brief(result), sort_keys=True), flush=True)
    (output_dir / "all_results.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )


def _brief(result: dict) -> dict:
    return {
        "seed": result["seed"],
        "methods": {
            method: {
                "brier": values["calibration"]["brier"],
                "aurc": values["calibration"]["aurc"],
                "alpha_0.20": values["risk_control"]["alpha_0.20"]["test"],
            }
            for method, values in result["methods"].items()
        },
    }


if __name__ == "__main__":
    main()
