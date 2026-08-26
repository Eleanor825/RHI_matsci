from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .features import feature_names_from_records
from .historical import MAIN_MATERIAL_TASKS, load_historical_tasks, split_historical_tasks
from .metrics import binary_metrics
from .risk_control import RiskControlledSelection, select_risk_controlled_threshold, selected_indices
from .schema import ActionRecord
from .scientific_gain import (
    EmpiricalUtilityNormalizer,
    GainConfig,
    GainTarget,
    fit_gain_distribution,
    gain_targets,
)
from .training import train_gate_with_features


DEFAULT_SEEDS = (1, 7, 13, 21, 42)


def run_sd_worth_experiment(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    alpha: float = 0.1,
    delta: float = 0.05,
    epochs: int = 90,
    gain_config: GainConfig = GainConfig(),
) -> dict[str, Any]:
    records_by_task = load_historical_tasks(data_dir, MAIN_MATERIAL_TASKS)
    runs = []
    for seed in seeds:
        task_splits = split_historical_tasks(records_by_task, seed=seed)
        splits = {
            split: _with_task_features(
                [record for task in MAIN_MATERIAL_TASKS for record in task_splits[task][split]]
            )
            for split in ("train", "feedback", "acceptance", "test")
        }
        normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
        targets = {
            split: gain_targets(records, normalizer, gain_config)
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
        reliability_feedback = reliability_gate.predict_proba(splits["feedback"])
        reliability_temperature = _calibrate_probability_temperature(
            reliability_feedback, [target.worthy for target in targets["feedback"]]
        )
        reliability_acceptance = _temperature_scale_probabilities(
            reliability_gate.predict_proba(splits["acceptance"]), reliability_temperature
        )
        reliability_selection = select_risk_controlled_threshold(
            reliability_acceptance,
            [target.worthy for target in targets["acceptance"]],
            alpha=alpha,
            delta=delta,
        )
        reliability_test = _temperature_scale_probabilities(
            reliability_gate.predict_proba(splits["test"]), reliability_temperature
        )

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
        worth_acceptance = _temperature_scale_probabilities(
            worth_gate.predict_proba(_replace_labels(splits["acceptance"], targets["acceptance"])),
            worth_temperature,
        )
        worth_selection = select_risk_controlled_threshold(
            worth_acceptance,
            [target.worthy for target in targets["acceptance"]],
            alpha=alpha,
            delta=delta,
        )
        worth_test = _temperature_scale_probabilities(
            worth_gate.predict_proba(_replace_labels(splits["test"], targets["test"])),
            worth_temperature,
        )

        gain_model = fit_gain_distribution(
            splits["train"],
            targets["train"],
            splits["feedback"],
            targets["feedback"],
            feature_names=feature_names,
            l2=0.1,
        )
        gain_acceptance = [
            prediction.action_worthiness for prediction in gain_model.predict(splits["acceptance"])
        ]
        gain_selection = select_risk_controlled_threshold(
            gain_acceptance,
            [target.worthy for target in targets["acceptance"]],
            alpha=alpha,
            delta=delta,
        )
        gain_test = [prediction.action_worthiness for prediction in gain_model.predict(splits["test"])]

        methods = {
            "static_reliability": _evaluate(
                splits["test"], targets["test"], reliability_test, reliability_selection
            ),
            "binary_worthiness": _evaluate(
                splits["test"], targets["test"], worth_test, worth_selection
            ),
            "sd_worth_gain_distribution": _evaluate(
                splits["test"], targets["test"], gain_test, gain_selection
            ),
        }
        runs.append(
            {
                "seed": seed,
                "sizes": {split: len(records) for split, records in splits.items()},
                "utility_normalizer": normalizer.to_json(),
                "feature_count": len(feature_names),
                "temperatures": {
                    "static_reliability": reliability_temperature,
                    "binary_worthiness": worth_temperature,
                    "sd_worth_gain_distribution": gain_model.temperature,
                },
                "methods": methods,
            }
        )

    summary = _summarize(runs)
    report = {
        "schema_version": "sd-worth-h1-v1",
        "method": "Source-Decomposed Worthiness H1 base without variance replicas",
        "confirmatory_protocol": "research/evl_iclr/experiments/h1_gain_risk/protocol.md",
        "data_dir": str(data_dir),
        "tasks": list(MAIN_MATERIAL_TASKS),
        "seeds": list(seeds),
        "alpha": alpha,
        "delta": delta,
        "gain_config": {
            "cost_weight": gain_config.cost_weight,
            "failure_harm_weight": gain_config.failure_harm_weight,
            "worthiness_margin": gain_config.worthiness_margin,
        },
        "runs": runs,
        "summary": summary,
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (destination / "RESULTS.md").write_text(_markdown(report))
    return report


def _with_task_features(records: list[ActionRecord]) -> list[ActionRecord]:
    tasks = sorted({record.benchmark for record in records})
    return [
        replace(
            record,
            features={
                **record.features,
                **{f"task::{task}": float(record.benchmark == task) for task in tasks},
            },
        )
        for record in records
    ]


def _replace_labels(records: list[ActionRecord], targets: list[GainTarget]) -> list[ActionRecord]:
    return [
        replace(record, label=target.worthy, utility=target.gain)
        for record, target in zip(records, targets)
    ]


def _evaluate(
    records: list[ActionRecord],
    targets: list[GainTarget],
    probabilities: list[float],
    selection: RiskControlledSelection,
) -> dict[str, Any]:
    labels = [target.worthy for target in targets]
    selected = selected_indices(probabilities, selection.threshold)
    selected_gains = [targets[index].gain for index in selected]
    selected_labels = [labels[index] for index in selected]
    oracle = sorted(range(len(targets)), key=lambda index: targets[index].gain, reverse=True)[: len(selected)]
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
    metrics = binary_metrics(labels, probabilities, selection.threshold)
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


def _calibrate_probability_temperature(probabilities: list[float], labels: list[int]) -> float:
    best_temperature = 1.0
    best_loss = float("inf")
    logits = [_logit(probability) for probability in probabilities]
    for step in range(2, 201):
        temperature = step / 20.0
        calibrated = [_sigmoid(value / temperature) for value in logits]
        loss = _log_loss(labels, calibrated)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _temperature_scale_probabilities(probabilities: list[float], temperature: float) -> list[float]:
    return [_sigmoid(_logit(probability) / temperature) for probability in probabilities]


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(runs[0]["methods"])
    metrics = [
        "coverage",
        "selective_risk",
        "mean_selected_gain",
        "net_gain_per_action",
        "oracle_gain_efficiency",
        "worst_regime_risk",
        "brier",
        "ece",
        "aurc",
    ]
    output = {}
    for method in methods:
        output[method] = {}
        for metric in metrics:
            values = [run["methods"][method]["test"][metric] for run in runs]
            output[method][metric] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        output[method]["calibration_risk_upper_bound"] = {
            "mean": mean(run["methods"][method]["selection"]["risk_upper_bound"] for run in runs),
        }
    return {"methods": output}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# H1 — Strict Net Gain and Risk-Controlled Worthiness",
        "",
        f"Seeds: `{report['seeds']}`; alpha: `{report['alpha']}`; delta: `{report['delta']}`.",
        "",
        "| Method | Coverage | Risk | Net gain/action | Gain efficiency | Worst-regime risk | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in report["summary"]["methods"].items():
        lines.append(
            f"| `{method}` | {metrics['coverage']['mean']:.4f} ± {metrics['coverage']['std']:.4f} | "
            f"{metrics['selective_risk']['mean']:.4f} ± {metrics['selective_risk']['std']:.4f} | "
            f"{metrics['net_gain_per_action']['mean']:.5f} ± {metrics['net_gain_per_action']['std']:.5f} | "
            f"{metrics['oracle_gain_efficiency']['mean']:.4f} ± {metrics['oracle_gain_efficiency']['std']:.4f} | "
            f"{metrics['worst_regime_risk']['mean']:.4f} | {metrics['brier']['mean']:.4f} | {metrics['ece']['mean']:.4f} |"
        )
    lines += [
        "",
        "The selection threshold is chosen on the independent acceptance split using a union-adjusted one-sided exact binomial risk upper bound. Test risk is reported but never used for threshold selection.",
    ]
    return "\n".join(lines) + "\n"


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _log_loss(labels: list[int], probabilities: list[float]) -> float:
    total = 0.0
    for label, probability in zip(labels, probabilities):
        probability = min(1.0 - 1e-12, max(1e-12, probability))
        total += -(label * math.log(probability) + (1 - label) * math.log(1.0 - probability))
    return total / max(1, len(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SD-Worth H1 on held-out materials regimes.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=90)
    args = parser.parse_args()
    report = run_sd_worth_experiment(
        args.data_dir,
        args.out_dir,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        alpha=args.alpha,
        delta=args.delta,
        epochs=args.epochs,
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
