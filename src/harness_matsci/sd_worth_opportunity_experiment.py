from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .features import feature_names_from_records
from .historical import MAIN_MATERIAL_TASKS, load_historical_tasks, split_historical_tasks
from .risk_control import binomial_upper_confidence, select_risk_controlled_grid
from .scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    fit_gain_distribution,
    gain_targets,
)
from .sd_worth_experiment import (
    _calibrate_probability_temperature,
    _replace_labels,
    _temperature_scale_probabilities,
    _with_task_features,
)
from .metrics import binary_metrics
from .training import train_gate_with_features


DEFAULT_SEEDS = (1, 7, 13, 21, 42)
DEFAULT_ALPHAS = (0.1, 0.2, 0.3)
DEFAULT_COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def run_sd_worth_opportunity_experiment(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    delta: float = 0.05,
    budget_fraction: float = 0.10,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
    epochs: int = 90,
    gain_config: GainConfig = GainConfig(),
) -> dict[str, Any]:
    records_by_task = load_historical_tasks(data_dir, MAIN_MATERIAL_TASKS)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        task_splits = split_historical_tasks(records_by_task, seed=seed)
        splits = {
            split: _with_task_features(
                [record for task in MAIN_MATERIAL_TASKS for record in task_splits[task][split]]
            )
            for split in ("train", "feedback", "acceptance", "test")
        }
        normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
        reservation = BudgetReservationValues.fit(
            splits["train"],
            normalizer,
            gain_config,
            budget_fraction=budget_fraction,
        )
        targets = {
            split: gain_targets(records, normalizer, gain_config, reservation)
            for split, records in splits.items()
        }
        feature_names = feature_names_from_records(splits["train"])

        reliability_gate = train_gate_with_features(
            splits["train"],
            splits["feedback"],
            feature_names=feature_names,
            epochs=epochs,
            learning_rate=0.08,
            l2=0.01,
            balance_benchmarks=True,
        )
        reliability_temperature = _calibrate_probability_temperature(
            reliability_gate.predict_proba(splits["feedback"]),
            [target.worthy for target in targets["feedback"]],
        )
        reliability_scores = {
            split: _temperature_scale_probabilities(
                reliability_gate.predict_proba(splits[split]), reliability_temperature
            )
            for split in ("acceptance", "test")
        }

        worth_train = _replace_labels(splits["train"], targets["train"])
        worth_feedback = _replace_labels(splits["feedback"], targets["feedback"])
        worth_gate = train_gate_with_features(
            worth_train,
            worth_feedback,
            feature_names=feature_names,
            epochs=epochs,
            learning_rate=0.08,
            l2=0.01,
            balance_benchmarks=True,
        )
        worth_temperature = _calibrate_probability_temperature(
            worth_gate.predict_proba(worth_feedback),
            [target.worthy for target in targets["feedback"]],
        )
        worth_scores = {
            split: _temperature_scale_probabilities(
                worth_gate.predict_proba(_replace_labels(splits[split], targets[split])),
                worth_temperature,
            )
            for split in ("acceptance", "test")
        }

        gain_model = fit_gain_distribution(
            splits["train"],
            targets["train"],
            splits["feedback"],
            targets["feedback"],
            feature_names=feature_names,
            l2=0.1,
        )
        gain_scores = {
            split: [
                prediction.action_worthiness
                for prediction in gain_model.predict(splits[split])
            ]
            for split in ("acceptance", "test")
        }
        task_binary_scores, task_binary_temperatures = _task_conditioned_binary_scores(
            splits, targets, feature_names, epochs
        )
        task_gain_scores, task_gain_temperatures = _task_conditioned_gain_scores(
            splits, targets, feature_names
        )
        scores = {
            "static_reliability": reliability_scores,
            "binary_budget_worthiness": worth_scores,
            "sd_worth_gain_distribution": gain_scores,
            "task_conditioned_binary_worthiness": task_binary_scores,
            "task_conditioned_gain_distribution": task_gain_scores,
        }
        methods_by_alpha: dict[str, Any] = {}
        acceptance_labels = [target.worthy for target in targets["acceptance"]]
        for alpha in alphas:
            methods_by_alpha[str(alpha)] = {}
            for method, method_scores in scores.items():
                selection = select_risk_controlled_grid(
                    method_scores["acceptance"],
                    acceptance_labels,
                    coverages=coverages,
                    alpha=alpha,
                    delta=delta,
                )
                methods_by_alpha[str(alpha)][method] = {
                    "acceptance_grid": _coverage_diagnostics(
                        method_scores["acceptance"], acceptance_labels, coverages, delta
                    ),
                    **_evaluate_at_coverage(
                        splits["test"], targets["test"], method_scores["test"], selection
                    ),
                }

        runs.append(
            {
                "seed": seed,
                "sizes": {split: len(records) for split, records in splits.items()},
                "utility_normalizer": normalizer.to_json(),
                "reservation_values": reservation.to_json(),
                "target_statistics": {
                    split: _target_statistics(splits[split], targets[split])
                    for split in splits
                },
                "feature_count": len(feature_names),
                "temperatures": {
                    "static_reliability": reliability_temperature,
                    "binary_budget_worthiness": worth_temperature,
                    "sd_worth_gain_distribution": gain_model.temperature,
                    "task_conditioned_binary_worthiness": task_binary_temperatures,
                    "task_conditioned_gain_distribution": task_gain_temperatures,
                },
                "methods_by_alpha": methods_by_alpha,
            }
        )

    report = {
        "schema_version": "sd-worth-h1.1-v1",
        "method": "Budget-conditioned Source-Decomposed Worthiness H1.1 without variance replicas",
        "confirmatory_protocol": "research/evl_iclr/experiments/h1_gain_risk/protocol_h1_1.md",
        "data_dir": str(data_dir),
        "tasks": list(MAIN_MATERIAL_TASKS),
        "seeds": list(seeds),
        "alphas": list(alphas),
        "delta": delta,
        "budget_fraction": budget_fraction,
        "coverage_grid": list(coverages),
        "gain_config": {
            "cost_weight": gain_config.cost_weight,
            "failure_harm_weight": gain_config.failure_harm_weight,
            "worthiness_margin": gain_config.worthiness_margin,
        },
        "runs": runs,
        "summary_by_alpha": _summarize(runs, alphas),
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (destination / "RESULTS.md").write_text(_markdown(report))
    return report


def _target_statistics(records, targets) -> dict[str, Any]:
    mismatch = sum(int(target.worthy != int(record.label)) for record, target in zip(records, targets))
    positive = sum(target.worthy for target in targets)
    by_task = {}
    for task in sorted({record.benchmark for record in records}):
        indices = [index for index, record in enumerate(records) if record.benchmark == task]
        task_positive = sum(targets[index].worthy for index in indices)
        task_mismatch = sum(
            int(targets[index].worthy != int(records[index].label)) for index in indices
        )
        by_task[task] = {
            "n": len(indices),
            "worthy_rate": task_positive / len(indices) if indices else 0.0,
            "success_mismatch_rate": task_mismatch / len(indices) if indices else 0.0,
            "mean_gain": mean(targets[index].gain for index in indices) if indices else 0.0,
        }
    return {
        "n": len(targets),
        "worthy_rate": positive / len(targets) if targets else 0.0,
        "success_mismatch_count": mismatch,
        "success_mismatch_rate": mismatch / len(targets) if targets else 0.0,
        "mean_gain": mean(target.gain for target in targets) if targets else 0.0,
        "by_task": by_task,
    }


def _task_conditioned_binary_scores(splits, targets, feature_names, epochs):
    output = {"acceptance": [0.0] * len(splits["acceptance"]), "test": [0.0] * len(splits["test"])}
    temperatures = {}
    for task in MAIN_MATERIAL_TASKS:
        train_indices = [index for index, record in enumerate(splits["train"]) if record.benchmark == task]
        feedback_indices = [index for index, record in enumerate(splits["feedback"]) if record.benchmark == task]
        task_train = _replace_labels(
            [splits["train"][index] for index in train_indices],
            [targets["train"][index] for index in train_indices],
        )
        task_feedback = _replace_labels(
            [splits["feedback"][index] for index in feedback_indices],
            [targets["feedback"][index] for index in feedback_indices],
        )
        gate = train_gate_with_features(
            task_train,
            task_feedback,
            feature_names=feature_names,
            epochs=epochs,
            learning_rate=0.08,
            l2=0.01,
            balance_benchmarks=False,
        )
        temperature = _calibrate_probability_temperature(
            gate.predict_proba(task_feedback),
            [record.label for record in task_feedback],
        )
        temperatures[task] = temperature
        for split in ("acceptance", "test"):
            indices = [index for index, record in enumerate(splits[split]) if record.benchmark == task]
            records = [splits[split][index] for index in indices]
            probabilities = _temperature_scale_probabilities(gate.predict_proba(records), temperature)
            for index, probability in zip(indices, probabilities):
                output[split][index] = probability
    return output, temperatures


def _task_conditioned_gain_scores(splits, targets, feature_names):
    output = {"acceptance": [0.0] * len(splits["acceptance"]), "test": [0.0] * len(splits["test"])}
    temperatures = {}
    for task in MAIN_MATERIAL_TASKS:
        train_indices = [index for index, record in enumerate(splits["train"]) if record.benchmark == task]
        feedback_indices = [index for index, record in enumerate(splits["feedback"]) if record.benchmark == task]
        model = fit_gain_distribution(
            [splits["train"][index] for index in train_indices],
            [targets["train"][index] for index in train_indices],
            [splits["feedback"][index] for index in feedback_indices],
            [targets["feedback"][index] for index in feedback_indices],
            feature_names=feature_names,
            l2=0.1,
        )
        temperatures[task] = model.temperature
        for split in ("acceptance", "test"):
            indices = [index for index, record in enumerate(splits[split]) if record.benchmark == task]
            records = [splits[split][index] for index in indices]
            probabilities = [prediction.action_worthiness for prediction in model.predict(records)]
            for index, probability in zip(indices, probabilities):
                output[split][index] = probability
    return output, temperatures


def _coverage_diagnostics(
    probabilities: list[float],
    labels: list[int],
    coverages: tuple[float, ...],
    delta: float,
) -> list[dict[str, float | int]]:
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    counts = sorted({max(1, min(len(order), math.ceil(len(order) * value))) for value in coverages})
    per_candidate_delta = delta / len(counts)
    diagnostics = []
    for count in counts:
        errors = sum(1 - labels[index] for index in order[:count])
        diagnostics.append(
            {
                "selected_count": count,
                "coverage": count / len(order),
                "empirical_risk": errors / count,
                "risk_upper_bound": binomial_upper_confidence(errors, count, per_candidate_delta),
            }
        )
    return diagnostics


def _evaluate_at_coverage(records, targets, probabilities, selection) -> dict[str, Any]:
    labels = [target.worthy for target in targets]
    selected_count = min(len(records), math.ceil(len(records) * selection.selected_fraction))
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    selected = order[:selected_count]
    selected_gains = [targets[index].gain for index in selected]
    selected_labels = [labels[index] for index in selected]
    oracle = sorted(range(len(targets)), key=lambda index: targets[index].gain, reverse=True)[:selected_count]
    oracle_gain = sum(targets[index].gain for index in oracle)
    total_selected_gain = sum(selected_gains)
    by_task = {}
    for task in sorted({record.benchmark for record in records}):
        task_indices = [index for index, record in enumerate(records) if record.benchmark == task]
        task_selected = [index for index in selected if records[index].benchmark == task]
        task_gain = sum(targets[index].gain for index in task_selected)
        by_task[task] = {
            "n": len(task_indices),
            "selected": len(task_selected),
            "coverage": len(task_selected) / len(task_indices) if task_indices else 0.0,
            "selective_risk": (
                sum(1 - labels[index] for index in task_selected) / len(task_selected)
                if task_selected else 0.0
            ),
            "net_gain_per_action": task_gain / len(task_indices) if task_indices else 0.0,
        }
    regime_risks = {}
    for regime in sorted({str(record.metadata.get("group_id", record.benchmark)) for record in records}):
        regime_selected = [
            index for index in selected
            if str(records[index].metadata.get("group_id", records[index].benchmark)) == regime
        ]
        if regime_selected:
            regime_risks[regime] = sum(1 - labels[index] for index in regime_selected) / len(regime_selected)
    metrics = binary_metrics(labels, probabilities, 0.5)
    return {
        "selection": selection.to_json(),
        "test": {
            "n": len(records),
            "selected": len(selected),
            "coverage": len(selected) / len(records) if records else 0.0,
            "selective_risk": 1.0 - mean(selected_labels) if selected_labels else 0.0,
            "mean_selected_gain": mean(selected_gains) if selected_gains else 0.0,
            "net_gain_per_action": total_selected_gain / len(records) if records else 0.0,
            "oracle_gain_efficiency": total_selected_gain / oracle_gain if oracle_gain > 0 else 0.0,
            "worst_regime_risk": max(regime_risks.values()) if regime_risks else 0.0,
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "aurc": metrics["aurc"],
            "by_task": by_task,
            "regime_risks": regime_risks,
        },
    }


def _summarize(runs: list[dict[str, Any]], alphas: tuple[float, ...]) -> dict[str, Any]:
    metrics = (
        "coverage",
        "selective_risk",
        "mean_selected_gain",
        "net_gain_per_action",
        "oracle_gain_efficiency",
        "worst_regime_risk",
        "brier",
        "ece",
        "aurc",
    )
    output: dict[str, Any] = {}
    for alpha in alphas:
        key = str(alpha)
        methods = sorted(runs[0]["methods_by_alpha"][key])
        output[key] = {}
        for method in methods:
            output[key][method] = {}
            for metric in metrics:
                values = [run["methods_by_alpha"][key][method]["test"][metric] for run in runs]
                output[key][method][metric] = {
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                }
            upper_values = [
                run["methods_by_alpha"][key][method]["selection"]["risk_upper_bound"]
                for run in runs
            ]
            output[key][method]["acceptance_risk_upper_bound"] = {
                "mean": mean(upper_values),
                "std": pstdev(upper_values) if len(upper_values) > 1 else 0.0,
                "values": upper_values,
            }
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SD-Worth H1.1: Budget-Conditioned Net Gain",
        "",
        f"Seeds: `{report['seeds']}`; action budget: `{report['budget_fraction']}`; delta: `{report['delta']}`.",
        "",
        "## Target sanity",
        "",
        "| Seed | Split | Worthy rate | Success mismatch | Mean gain |",
        "|---:|---|---:|---:|---:|",
    ]
    for run in report["runs"]:
        for split, stats in run["target_statistics"].items():
            lines.append(
                f"| {run['seed']} | {split} | {stats['worthy_rate']:.4f} | "
                f"{stats['success_mismatch_rate']:.4f} | {stats['mean_gain']:.4f} |"
            )
    for alpha, methods in report["summary_by_alpha"].items():
        lines.extend(
            [
                "",
                f"## Risk-controlled test results at alpha={alpha}",
                "",
                "| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, metrics in methods.items():
            lines.append(
                f"| {method} | {metrics['coverage']['mean']:.4f} | "
                f"{metrics['selective_risk']['mean']:.4f} | "
                f"{metrics['net_gain_per_action']['mean']:.4f} | "
                f"{metrics['brier']['mean']:.4f} | {metrics['ece']['mean']:.4f} | "
                f"{metrics['aurc']['mean']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run budget-conditioned SD-Worth H1.1.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=90)
    args = parser.parse_args()
    report = run_sd_worth_opportunity_experiment(
        args.data_dir,
        args.out_dir,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        alphas=tuple(float(value) for value in args.alphas.split(",") if value),
        delta=args.delta,
        budget_fraction=args.budget_fraction,
        epochs=args.epochs,
    )
    print(json.dumps(report["summary_by_alpha"], indent=2))


if __name__ == "__main__":
    main()
