from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from harness_matsci.historical import (
    MAIN_MATERIAL_TASKS,
    grouped_four_way_fold,
    load_historical_tasks,
)
from harness_matsci.hurdle_gain_calibration import (
    HurdleGainObservation,
    fit_hurdle_gain_calibrator,
)
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    gain_targets,
)
from harness_matsci.sequential_tool_benchmark import RealScientificToolEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    result = run_pilot(args.raw_data_dir, fold=args.fold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"fold_{args.fold}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / f"FOLD_{args.fold}.md").write_text(
        _markdown(result), encoding="utf-8"
    )


def run_pilot(raw_data_dir: str, *, fold: int) -> dict:
    records_by_task = load_historical_tasks(raw_data_dir, MAIN_MATERIAL_TASKS)
    splits_by_task = {
        task: grouped_four_way_fold(
            records,
            fold_index=fold,
            n_folds=5,
            train_fraction=0.50,
            feedback_fraction=0.15,
            acceptance_fraction=0.20,
        )
        for task, records in records_by_task.items()
    }
    splits = {
        split: [
            record
            for task in MAIN_MATERIAL_TASKS
            for record in splits_by_task[task][split]
        ]
        for split in ("train", "feedback", "test")
    }
    normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
    config = GainConfig(utility_mode="continuous_realized")
    reservations = BudgetReservationValues.fit(
        splits["train"],
        normalizer,
        config,
        budget_fraction=0.10,
        outside_option=0.0,
        strategy="potential_success_slot_value",
    )
    targets_by_split = {
        split: gain_targets(records, normalizer, config, reservations)
        for split, records in splits.items()
    }
    target_by_id = {
        target.record_id: target
        for targets in targets_by_split.values()
        for target in targets
    }
    environment = RealScientificToolEnvironment.fit(
        raw_data_dir,
        splits["train"],
        environment_seed=1729 + fold,
    )
    observations = {
        split: _tool_observations(
            environment,
            records,
            target_by_id,
            run_seed=2729 + fold,
        )
        for split, records in splits.items()
        if split != "train"
    }
    methods = {}
    for use_source_variance, source_name in ((False, "no_source"), (True, "source")):
        fitted = fit_hurdle_gain_calibrator(
            observations["feedback"],
            use_source_variance=use_source_variance,
            beta=0.10,
            tail_weight=0.50,
        )
        predictions = fitted.predict(observations["test"])
        methods[source_name] = {
            "model": fitted.to_json(),
            "tasks": {
                task: _task_result(
                    [
                        prediction
                        for prediction in predictions
                        if prediction.benchmark == task
                    ],
                    [
                        target_by_id[prediction.record_id].gain
                        for prediction in predictions
                        if prediction.benchmark == task
                    ],
                    [
                        target_by_id[prediction.record_id].worthy
                        for prediction in predictions
                        if prediction.benchmark == task
                    ],
                )
                for task in MAIN_MATERIAL_TASKS
            },
        }
    return {
        "experiment": "h14_tail_risk_real_tool_ranking_pilot",
        "development_only": True,
        "fold": fold,
        "tool_cost_per_acquired_action": 0.002,
        "methods": methods,
    }


def _tool_observations(
    environment: RealScientificToolEnvironment,
    records: list,
    target_by_id: dict,
    *,
    run_seed: int,
) -> list[HurdleGainObservation]:
    output = []
    for record in records:
        margins = []
        for evidence_replica in (101, 211, 307):
            observation = environment.observe(
                record,
                stage="verification",
                evidence_replica=evidence_replica,
                run_seed=run_seed,
            )
            margins.append(float(observation.record.features["tool_estimated_margin"]))
        output.append(
            HurdleGainObservation(
                record_id=record.record_id,
                benchmark=record.benchmark,
                mean_gain=mean(margins),
                internal_variance=0.0,
                external_variance=_population_variance(margins),
                gain=target_by_id[record.record_id].gain,
            )
        )
    return output


def _task_result(predictions: list, gains: list[float], labels: list[int]) -> dict:
    score_families = {
        "p_positive_gain": [prediction.p_positive_gain for prediction in predictions],
        "mean_gain": [prediction.mean_gain for prediction in predictions],
        "tail_risk_score": [prediction.tail_risk_score for prediction in predictions],
    }
    result = {
        name: {
            key: value
            for key, value in _selection_grid(scores, gains, labels).items()
        }
        for name, scores in score_families.items()
    }
    result["zero_threshold"] = {
            name: _zero_threshold_summary(scores, gains, labels)
            for name, scores in score_families.items()
    }
    return result


def _zero_threshold_summary(
    scores: list[float], gains: list[float], labels: list[int]
) -> dict:
    selected = [index for index, score in enumerate(scores) if score > 0.0]
    if not selected:
        return {
            "selected_count": 0,
            "precision": 0.0,
            "selected_mean_gain": 0.0,
            "population_gain_after_selected_only_tool_cost": 0.0,
        }
    selected_gain = sum(gains[index] for index in selected)
    return {
        "selected_count": len(selected),
        "precision": mean(labels[index] for index in selected),
        "selected_mean_gain": selected_gain / len(selected),
        "population_gain_after_selected_only_tool_cost": (
            selected_gain - 0.002 * len(selected)
        )
        / len(scores),
    }


def _selection_grid(scores: list[float], gains: list[float], labels: list[int]) -> dict:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    output = {}
    for fraction in (0.005, 0.01, 0.02, 0.05, 0.10):
        count = max(1, round(fraction * len(scores)))
        selected = order[:count]
        selected_gain = sum(gains[index] for index in selected)
        population_gain = selected_gain / len(scores)
        output[f"top_{fraction:.3f}"] = {
            "selected_count": count,
            "precision": mean(labels[index] for index in selected),
            "selected_mean_gain": selected_gain / count,
            "population_gain_before_tool_cost": population_gain,
            "population_gain_after_selected_only_tool_cost": population_gain
            - 0.002 * count / len(scores),
        }
    return output


def _population_variance(values: list[float]) -> float:
    center = mean(values)
    return mean((value - center) ** 2 for value in values)


def _markdown(result: dict) -> str:
    lines = [
        "# H14 Tail-Risk Real-Tool Ranking Pilot",
        "",
        "Development-only result on consumed fold 0. Tool acquisition is assumed only",
        "for selected actions in the cost column; learned VoE is not yet included.",
        "",
        "| source | task | score | best prefix | precision | selected gain | population gain after cost |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for source_name, source_result in result["methods"].items():
        for task, task_result in source_result["tasks"].items():
            for score_name, grid in task_result.items():
                if score_name == "zero_threshold":
                    continue
                prefix_name, prefix = max(
                    grid.items(),
                    key=lambda item: item[1]["population_gain_after_selected_only_tool_cost"],
                )
                lines.append(
                    f"| `{source_name}` | `{task}` | `{score_name}` | `{prefix_name}` | "
                    f"{prefix['precision']:.4f} | {prefix['selected_mean_gain']:+.4f} | "
                    f"{prefix['population_gain_after_selected_only_tool_cost']:+.6f} |"
                )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
