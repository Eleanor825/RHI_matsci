from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


METHOD_FIELDS = {
    "factorized_worthiness": lambda row: float(row["factorized"]["action_worthiness"]),
    "direct_llm": lambda row: float(row["direct_probability"]),
    "static_reliability": lambda row: float(row["static_reliability_probability"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    report = summarize(
        json.loads(Path(args.results).read_text(encoding="utf-8")),
        samples=args.samples,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_bootstrap.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "paired_bootstrap.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["comparisons"], sort_keys=True))


def summarize(result, *, samples, seed):
    rows = result["test_predictions"]
    observed = {
        method: _metrics(rows, scorer)
        for method, scorer in METHOD_FIELDS.items()
    }
    rng = random.Random(seed)
    bootstrap = {
        baseline: {"brier_gain": [], "log_loss_gain": [], "top10_gain_advantage": []}
        for baseline in ("direct_llm", "static_reliability")
    }
    by_task = {
        task: [row for row in rows if row["benchmark"] == task]
        for task in sorted({row["benchmark"] for row in rows})
    }
    for _ in range(samples):
        sampled = [
            rng.choice(task_rows)
            for task_rows in by_task.values()
            for _ in range(len(task_rows))
        ]
        ours = _metrics(sampled, METHOD_FIELDS["factorized_worthiness"])
        for baseline in bootstrap:
            other = _metrics(sampled, METHOD_FIELDS[baseline])
            bootstrap[baseline]["brier_gain"].append(other["brier"] - ours["brier"])
            bootstrap[baseline]["log_loss_gain"].append(
                other["log_loss"] - ours["log_loss"]
            )
            bootstrap[baseline]["top10_gain_advantage"].append(
                ours["top10_population_net_gain"]
                - other["top10_population_net_gain"]
            )
    comparisons = {}
    for baseline, values in bootstrap.items():
        comparisons[baseline] = {
            metric: _interval(samples_for_metric)
            for metric, samples_for_metric in values.items()
        }
    return {
        "schema_version": "factorized-confirmatory-paired-bootstrap-v1",
        "bootstrap_samples": samples,
        "seed": seed,
        "test_record_count": len(rows),
        "observed": observed,
        "comparisons": comparisons,
    }


def _metrics(rows, scorer):
    labels = [int(row["actual_net_gain"] > 0.0) for row in rows]
    probabilities = [_clip(scorer(row)) for row in rows]
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(rows)
    log_loss = -sum(
        label * math.log(probability) + (1-label) * math.log(1-probability)
        for probability, label in zip(probabilities, labels)
    ) / len(rows)
    selected = []
    for task in sorted({row["benchmark"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["benchmark"] == task]
        count = max(1, math.ceil(0.10 * len(indices)))
        selected.extend(sorted(indices, key=lambda index: (-probabilities[index], index))[:count])
    gains = [float(rows[index]["actual_net_gain"]) for index in selected]
    return {
        "brier": brier,
        "log_loss": log_loss,
        "top10_selected_count": len(selected),
        "top10_selective_risk": sum(gain <= 0.0 for gain in gains) / len(gains),
        "top10_selected_mean_gain": sum(gains) / len(gains),
        "top10_population_net_gain": sum(gains) / len(rows),
    }


def _interval(values):
    ordered = sorted(values)
    lower = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered)-1, math.ceil(0.975 * (len(ordered) - 1)))]
    return {
        "mean": sum(values) / len(values),
        "ci_95": [lower, upper],
        "probability_not_positive": sum(value <= 0.0 for value in values) / len(values),
    }


def _markdown(report):
    lines = ["# Paired stratified bootstrap", "", f"Test records: {report['test_record_count']}", ""]
    for baseline, comparisons in report["comparisons"].items():
        lines.append(f"## Ours vs {baseline}")
        for metric, values in comparisons.items():
            lines.append(
                f"- `{metric}`: mean={values['mean']:.6f}, "
                f"95% CI=[{values['ci_95'][0]:.6f}, {values['ci_95'][1]:.6f}], "
                f"P(diff<=0)={values['probability_not_positive']:.4f}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _clip(value):
    return min(1.0 - 1e-9, max(1e-9, float(value)))


if __name__ == "__main__":
    main()
