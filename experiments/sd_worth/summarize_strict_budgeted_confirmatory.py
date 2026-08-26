from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from harness_matsci.budgeted_gain_policy import (
    BudgetedGainPolicy,
    evaluate_budgeted_gain_policy,
)


PRIMARY = "factorized_conformal_lower_gain"
BASELINES = (
    "factorized_expected_gain",
    "direct_llm_judge",
    "llm_component_gain",
    "static_reliability",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=4729)
    args = parser.parse_args()
    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = summarize(result, samples=args.samples, seed=args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_budgeted_bootstrap.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "paired_budgeted_bootstrap.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["comparisons"], sort_keys=True))


def summarize(result, *, samples, seed):
    rows = result["test_predictions"]
    if not rows:
        raise ValueError("full confirmatory test predictions are required")
    policies = {
        method: BudgetedGainPolicy(**result["policies"][method])
        for method in (PRIMARY, *BASELINES)
    }
    observed = {
        method: _evaluate(rows, method, policies[method])
        for method in policies
    }
    observed["oracle"] = _oracle(rows, policies[PRIMARY].budget_fraction)
    task_observed = {
        method: _task_breakdown(rows, method, policies[method])
        for method in policies
    }
    task_observed["oracle"] = _task_oracle_breakdown(
        rows,
        policies[PRIMARY].budget_fraction,
    )
    rng = random.Random(seed)
    by_task = {
        task: [row for row in rows if row["benchmark"] == task]
        for task in sorted({row["benchmark"] for row in rows})
    }
    draws = {
        baseline: {
            "total_gain_advantage": [],
            "mean_gain_advantage": [],
            "risk_reduction": [],
            "population_gain_advantage": [],
        }
        for baseline in BASELINES
    }
    oracle_efficiency = []
    for _ in range(samples):
        sample = [
            rng.choice(task_rows)
            for task_rows in by_task.values()
            for _ in range(len(task_rows))
        ]
        ours = _evaluate(sample, PRIMARY, policies[PRIMARY])
        oracle = _oracle(sample, policies[PRIMARY].budget_fraction)
        oracle_efficiency.append(
            ours["selected_total_net_gain"] / oracle["selected_total_net_gain"]
            if oracle["selected_total_net_gain"] > 0.0
            else 0.0
        )
        for baseline in BASELINES:
            other = _evaluate(sample, baseline, policies[baseline])
            draws[baseline]["total_gain_advantage"].append(
                ours["selected_total_net_gain"] - other["selected_total_net_gain"]
            )
            draws[baseline]["mean_gain_advantage"].append(
                ours["selected_mean_net_gain"] - other["selected_mean_net_gain"]
            )
            draws[baseline]["risk_reduction"].append(
                other["selective_risk"] - ours["selective_risk"]
            )
            draws[baseline]["population_gain_advantage"].append(
                ours["population_net_gain"] - other["population_net_gain"]
            )
    return {
        "schema_version": "strict-budgeted-confirmatory-bootstrap-v1",
        "samples": samples,
        "seed": seed,
        "test_record_count": len(rows),
        "observed": observed,
        "task_observed": task_observed,
        "oracle_efficiency": _interval(oracle_efficiency),
        "comparisons": {
            baseline: {
                metric: _interval(values) for metric, values in metrics.items()
            }
            for baseline, metrics in draws.items()
        },
    }


def _evaluate(rows, method, policy):
    return evaluate_budgeted_gain_policy(
        policy,
        [float(row["scores"][method]) for row in rows],
        [float(row["actual_strict_net_gain"]) for row in rows],
    )


def _task_breakdown(rows, method, policy):
    selected = set(
        policy.select([float(row["scores"][method]) for row in rows])
    )
    output = {}
    for task in sorted({row["benchmark"] for row in rows}):
        task_indices = [
            index for index, row in enumerate(rows) if row["benchmark"] == task
        ]
        selected_indices = [index for index in task_indices if index in selected]
        selected_gains = [
            float(rows[index]["actual_strict_net_gain"])
            for index in selected_indices
        ]
        errors = sum(gain <= 0.0 for gain in selected_gains)
        output[task] = {
            "population_count": len(task_indices),
            "selected_count": len(selected_indices),
            "coverage": len(selected_indices) / len(task_indices),
            "selection_share": len(selected_indices) / len(selected) if selected else 0.0,
            "errors": errors,
            "selective_risk": errors / len(selected_indices) if selected_indices else None,
            "selected_mean_net_gain": (
                sum(selected_gains) / len(selected_gains) if selected_gains else None
            ),
            "selected_total_net_gain": sum(selected_gains),
        }
    return output


def _task_oracle_breakdown(rows, budget_fraction):
    output = {}
    for task in sorted({row["benchmark"] for row in rows}):
        task_rows = [row for row in rows if row["benchmark"] == task]
        output[task] = _oracle(task_rows, budget_fraction)
    return output


def _oracle(rows, budget_fraction):
    gains = [float(row["actual_strict_net_gain"]) for row in rows]
    policy = BudgetedGainPolicy(budget_fraction, minimum_score=0.0)
    return evaluate_budgeted_gain_policy(policy, gains, gains)


def _interval(values):
    ordered = sorted(values)
    lower = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return {
        "mean": sum(values) / len(values),
        "ci_95": [lower, upper],
        "probability_not_positive": sum(value <= 0.0 for value in values) / len(values),
    }


def _markdown(report):
    lines = [
        "# Strict budgeted confirmatory bootstrap",
        "",
        f"Test records: {report['test_record_count']}",
        f"Bootstrap samples: {report['samples']}",
        "",
        "## Observed",
    ]
    for method, metrics in report["observed"].items():
        lines.append(
            f"- `{method}`: selected={metrics['selected_count']}, "
            f"risk={metrics['selective_risk']:.6f}, "
            f"mean_gain={metrics['selected_mean_net_gain']:.6f}, "
            f"total_gain={metrics['selected_total_net_gain']:.6f}"
        )
    lines.extend(["", "## Paired task-stratified comparisons"])
    for baseline, metrics in report["comparisons"].items():
        lines.append(f"### Ours vs `{baseline}`")
        for metric, values in metrics.items():
            lines.append(
                f"- `{metric}`: mean={values['mean']:.6f}, "
                f"95% CI=[{values['ci_95'][0]:.6f}, {values['ci_95'][1]:.6f}], "
                f"P(diff<=0)={values['probability_not_positive']:.6f}"
            )
    lines.extend(["", "## Task-wise selection audit"])
    for method, tasks in report["task_observed"].items():
        lines.append(f"### `{method}`")
        for task, metrics in tasks.items():
            risk = metrics.get("selective_risk")
            mean_gain = metrics.get("selected_mean_net_gain")
            risk_text = "NA" if risk is None else f"{risk:.6f}"
            mean_text = "NA" if mean_gain is None else f"{mean_gain:.6f}"
            selection_share = metrics.get("selection_share")
            selection_share_text = (
                "NA" if selection_share is None else f"{selection_share:.6f}"
            )
            lines.append(
                f"- `{task}`: selected={metrics['selected_count']}/"
                f"{metrics.get('population_count', metrics['selected_count'])}, "
                f"coverage={metrics['coverage']:.6f}, "
                f"selection_share={selection_share_text}, risk={risk_text}, "
                f"mean_gain={mean_text}, "
                f"total_gain={metrics['selected_total_net_gain']:.6f}"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
