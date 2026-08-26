from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev


PRIMARY = "task_sparse_tool_margin_continuous_sd_worth"
NO_SOURCE = "task_sparse_tool_margin_continuous_mean"
STATIC = "base_mean"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    summary = summarize(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary["confirmatory_criteria"], indent=2, sort_keys=True))


def summarize(results: list[dict]) -> dict:
    rows = []
    for result in results:
        primary = _test(result, PRIMARY)
        no_source = _test(result, NO_SOURCE)
        static = _test(result, STATIC)
        rows.append(
            {
                "seed": result["seed"],
                "audit_passed": result["nontraining_outcome_invariance_audit"]["passed"],
                "coverage": primary["coverage"],
                "selective_risk": primary["selective_risk"],
                "selected_mean_gain": primary["selected_mean_gain"],
                "population_net_gain": primary["population_net_gain"],
                "acquisition_fraction": primary["acquisition_fraction"],
                "selected_tasks": sorted(primary["task_risks"]),
                "delta_vs_no_source": primary["population_net_gain"]
                - no_source["population_net_gain"],
                "delta_vs_static": primary["population_net_gain"]
                - static["population_net_gain"],
            }
        )
    source_deltas = [row["delta_vs_no_source"] for row in rows]
    static_deltas = [row["delta_vs_static"] for row in rows]
    criteria = {
        "audit_pass_seeds": sum(row["audit_passed"] for row in rows),
        "nonzero_coverage_seeds": sum(row["coverage"] > 0.0 for row in rows),
        "risk_at_most_0.20_seeds": sum(row["selective_risk"] <= 0.20 for row in rows),
        "positive_selected_gain_seeds": sum(row["selected_mean_gain"] > 0.0 for row in rows),
        "positive_population_net_gain_seeds": sum(
            row["population_net_gain"] > 0.0 for row in rows
        ),
        "beats_no_source_seeds": sum(value > 0.0 for value in source_deltas),
        "positive_mean_delta_vs_no_source": mean(source_deltas) > 0.0,
        "positive_mean_delta_vs_static": mean(static_deltas) > 0.0,
    }
    criteria["h10_confirmatory_pass"] = (
        criteria["audit_pass_seeds"] == len(rows)
        and criteria["nonzero_coverage_seeds"] >= 4
        and criteria["risk_at_most_0.20_seeds"] >= 4
        and criteria["positive_selected_gain_seeds"] >= 4
        and criteria["positive_population_net_gain_seeds"] >= 4
    )
    return {
        "experiment": "h10_real_scientific_tools_five_seeds",
        "primary_method": PRIMARY,
        "no_source_ablation": NO_SOURCE,
        "static_baseline": STATIC,
        "rows": rows,
        "aggregate": {
            metric: _aggregate([row[metric] for row in rows])
            for metric in (
                "coverage",
                "selective_risk",
                "selected_mean_gain",
                "population_net_gain",
                "acquisition_fraction",
                "delta_vs_no_source",
                "delta_vs_static",
            )
        },
        "paired_inference": {
            "source_bootstrap_mean_ci_95": _bootstrap_mean_ci(source_deltas, 20000, 101),
            "source_wilcoxon": _wilcoxon(source_deltas),
            "static_bootstrap_mean_ci_95": _bootstrap_mean_ci(static_deltas, 20000, 103),
            "static_wilcoxon": _wilcoxon(static_deltas),
        },
        "confirmatory_criteria": criteria,
    }


def _test(result: dict, method: str) -> dict:
    return result["methods"][method]["risk_control"]["alpha_0.20"]["test"]


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "population_std": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return estimates[math.floor(0.025 * samples)], estimates[math.floor(0.975 * samples)]


def _wilcoxon(values: list[float]) -> dict[str, float | None]:
    if not any(value != 0.0 for value in values):
        return {"statistic": 0.0, "pvalue": 1.0}
    from scipy.stats import wilcoxon

    result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def render(summary: dict) -> str:
    lines = [
        "# H10 Real Scientific Tool Results",
        "",
        "Primary: `task_sparse_tool_margin_continuous_sd_worth`; exact binary selective risk at `alpha=0.20`; normalized verification cost `0.002`.",
        "",
        "| seed | audit | coverage | risk | selected gain | acquisition | population net gain | delta vs no-source | tasks selected |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['seed']} | {row['audit_passed']} | {row['coverage']:.4f} | "
            f"{row['selective_risk']:.4f} | {row['selected_mean_gain']:.4f} | "
            f"{row['acquisition_fraction']:.4f} | {row['population_net_gain']:.6f} | "
            f"{row['delta_vs_no_source']:+.6f} | `{','.join(row['selected_tasks'])}` |"
        )
    lines.extend(["", "## Aggregate", ""])
    aggregate = summary["aggregate"]
    for metric in (
        "coverage",
        "selective_risk",
        "selected_mean_gain",
        "acquisition_fraction",
        "population_net_gain",
        "delta_vs_no_source",
        "delta_vs_static",
    ):
        lines.append(
            f"- Mean {metric.replace('_', ' ')}: `{aggregate[metric]['mean']:.6f}`; "
            f"median `{aggregate[metric]['median']:.6f}`."
        )
    lines.extend(["", "## Confirmatory audit", ""])
    for key, value in summary["confirmatory_criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Paired inference", ""])
    for key, value in summary["paired_inference"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
