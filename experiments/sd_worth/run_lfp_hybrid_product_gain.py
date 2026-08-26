from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.hybrid_product_gain import (
    build_hybrid_product_states,
    evaluate_hybrid_product_gain,
    load_llm_probability_caches,
)
from harness_matsci.lfp_crossed_gain import (
    CrossedPairwiseGainModel,
    apply_source_fuser,
    fit_candidate_excluded_source_fuser,
)
from harness_matsci.schema import ActionRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-root", required=True)
    parser.add_argument("--calibration-cache-dir", required=True)
    parser.add_argument("--test-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        default="gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra",
    )
    parser.add_argument("--replicas", type=int, default=12)
    args = parser.parse_args()

    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if len(models) < 2:
        raise ValueError("hybrid evaluation requires at least two internal models")
    folds_root = Path(args.folds_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_paths = {
        model: Path(args.calibration_cache_dir) / f"{model}.json" for model in models
    }
    test_paths = {model: Path(args.test_cache_dir) / f"{model}.json" for model in models}

    fold_reports = []
    for fold_dir in sorted(path for path in folds_root.glob("fold_*") if path.is_dir()):
        report = _run_fold(
            fold_dir=fold_dir,
            models=models,
            calibration_paths=calibration_paths,
            test_paths=test_paths,
            replicas=args.replicas,
        )
        fold_reports.append(report)
        (output_dir / f"{fold_dir.name}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    aggregate = _aggregate(fold_reports, models)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate["summary"], sort_keys=True))


def _run_fold(
    *,
    fold_dir: Path,
    models: tuple[str, ...],
    calibration_paths: dict[str, Path],
    test_paths: dict[str, Path],
    replicas: int,
) -> dict[str, object]:
    splits = {
        split: _load(fold_dir / f"{split}.jsonl")
        for split in ("train", "feedback", "acceptance", "test")
    }
    fuser = fit_candidate_excluded_source_fuser(splits["train"])
    value_model = CrossedPairwiseGainModel(replicas=replicas).fit(splits["train"])
    raw_states = {
        split: apply_source_fuser(value_model.predict(splits[split]), fuser)
        for split in ("feedback", "acceptance", "test")
    }
    source_weights = fuser.parameters_by_task[
        "lfp_real_multitask_shared_source_fusion"
    ].source_weights
    probabilities = {
        "feedback": load_llm_probability_caches(
            splits["feedback"], model_cache_paths=calibration_paths
        ),
        "acceptance": load_llm_probability_caches(
            splits["acceptance"], model_cache_paths=calibration_paths
        ),
        "test": load_llm_probability_caches(
            splits["test"], model_cache_paths=test_paths
        ),
    }
    hybrid_states = {
        split: build_hybrid_product_states(
            splits[split],
            [state.to_json() for state in raw_states[split]],
            probabilities[split],
            source_weights,
        )
        for split in ("feedback", "acceptance", "test")
    }
    evaluation = evaluate_hybrid_product_gain(
        feedback=hybrid_states["feedback"],
        acceptance=hybrid_states["acceptance"],
        test=hybrid_states["test"],
        model_names=models,
    )
    return {
        "fold": fold_dir.name,
        "models": list(models),
        "source_fuser": fuser.to_json(),
        "split_counts": {split: len(records) for split, records in splits.items()},
        "evaluation": evaluation,
        "states": {
            split: [state.to_json() for state in states]
            for split, states in hybrid_states.items()
        },
    }


def _aggregate(
    fold_reports: list[dict[str, object]], models: tuple[str, ...]
) -> dict[str, object]:
    evaluations = [report["evaluation"] for report in fold_reports]
    summary = {
        "folds": len(fold_reports),
        "hybrid": _mean_metrics([row["hybrid_metrics"] for row in evaluations]),
        "direct_llm_ensemble": _mean_metrics(
            [row["direct_llm_ensemble_metrics"] for row in evaluations]
        ),
        "direct_models": {
            model: _mean_metrics(
                [row["direct_llm_model_metrics"][model] for row in evaluations]
            )
            for model in models
        },
        "component_ablations": {
            mode: _mean_metrics(
                [row["component_ablation_metrics"][mode] for row in evaluations]
            )
            for mode in ("mean_only", "internal_only", "internal_external", "full")
        },
        "certified_folds": sum(row["certificate"]["certified"] for row in evaluations),
        "mean_conformal_coverage": _mean(
            row["test_conformal_coverage"] for row in evaluations
        ),
        "mean_deployed_coverage": _mean(
            row["deployed"]["coverage"] for row in evaluations
        ),
        "mean_population_net_gain": _mean(
            row["deployed"]["population_net_gain"] for row in evaluations
        ),
    }
    return {"models": list(models), "summary": summary, "fold_reports": fold_reports}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0]
    return {key: _mean(row[key] for row in rows) for key in keys}


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _load(path: Path) -> list[ActionRecord]:
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
