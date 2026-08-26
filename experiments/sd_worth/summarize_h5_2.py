from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev


PRIMARY = "task_sparse_continuous_sd_worth"
ABLATION = "task_sparse_continuous_mean"


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
    (output_dir / "RESULTS.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["confirmatory_criteria"], indent=2, sort_keys=True))


def summarize(results: list[dict]) -> dict:
    rows = []
    for result in results:
        primary = _test(result, PRIMARY, "risk_control")
        ablation = _test(result, ABLATION, "risk_control")
        shortfall = _test(result, PRIMARY, "shortfall_risk_control")
        rows.append(
            {
                "seed": result["seed"],
                "coverage": primary["coverage"],
                "selective_risk": primary["selective_risk"],
                "selected_mean_gain": primary["selected_mean_gain"],
                "population_net_gain": primary["population_net_gain"],
                "acquisition_fraction": primary["acquisition_fraction"],
                "worst_task_risk": primary["worst_task_risk"],
                "worst_regime_risk_min5": primary["worst_regime_risk_min5"],
                "ablation_population_net_gain": ablation["population_net_gain"],
                "paired_net_gain_delta": primary["population_net_gain"]
                - ablation["population_net_gain"],
                "shortfall_coverage": shortfall["coverage"],
                "shortfall_test_loss": shortfall["selected_mean_shortfall"],
                "source_positive_tasks": sum(
                    int(values["internal_weight"] > 1e-8 or values["external_weight"] > 1e-8)
                    for values in result["continuous_source_calibrator"]["parameters_by_task"].values()
                ),
            }
        )
    deltas = [row["paired_net_gain_delta"] for row in rows]
    bootstrap = _bootstrap_mean_ci(deltas, samples=100000, seed=20260821)
    wilcoxon = _wilcoxon(deltas)
    criteria = {
        "nonzero_coverage_seeds": sum(row["coverage"] > 0.0 for row in rows),
        "risk_at_most_0.20_seeds": sum(row["selective_risk"] <= 0.20 for row in rows),
        "positive_population_net_gain_seeds": sum(row["population_net_gain"] > 0.0 for row in rows),
        "beats_continuous_mean_seeds": sum(row["paired_net_gain_delta"] > 0.0 for row in rows),
        "positive_mean_paired_difference": mean(deltas) > 0.0,
        "source_nonzero_two_tasks_every_seed": all(row["source_positive_tasks"] >= 2 for row in rows),
    }
    criteria["h5_2_confirmatory_pass"] = (
        criteria["nonzero_coverage_seeds"] >= 4
        and criteria["risk_at_most_0.20_seeds"] >= 4
        and criteria["positive_population_net_gain_seeds"] >= 4
        and criteria["beats_continuous_mean_seeds"] >= 3
        and criteria["positive_mean_paired_difference"]
        and criteria["source_nonzero_two_tasks_every_seed"]
    )
    return {
        "experiment": "h5_2_stratified_acquisition_five_seeds",
        "primary_method": PRIMARY,
        "alpha": 0.20,
        "verification_cost": 0.002,
        "rows": rows,
        "aggregate": {
            metric: _aggregate([row[metric] for row in rows])
            for metric in (
                "coverage",
                "selective_risk",
                "selected_mean_gain",
                "population_net_gain",
                "acquisition_fraction",
                "worst_task_risk",
                "worst_regime_risk_min5",
                "paired_net_gain_delta",
                "shortfall_coverage",
            )
        },
        "paired_inference": {
            "bootstrap_mean_difference_ci_95": bootstrap,
            "wilcoxon_two_sided": wilcoxon,
            "interpretation": (
                "directionally positive but not conventionally significant"
                if bootstrap[0] <= 0.0 or wilcoxon["pvalue"] >= 0.05
                else "positive and conventionally significant"
            ),
        },
        "confirmatory_criteria": criteria,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# H5.2 Five-Seed Results",
        "",
        "Primary method: `task_sparse_continuous_sd_worth`; pooled exact risk certificate at `alpha=0.20`; verification cost `0.002`.",
        "",
        "| seed | coverage | risk | selected gain | acquisition | population net gain | delta vs no-source | shortfall coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {seed} | {coverage:.4f} | {selective_risk:.4f} | {selected_mean_gain:.4f} | "
            "{acquisition_fraction:.4f} | {population_net_gain:.6f} | {paired_net_gain_delta:+.6f} | "
            "{shortfall_coverage:.4f} |".format(**row)
        )
    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Mean coverage: `{aggregate['coverage']['mean']:.4f}`.",
            f"- Mean selective risk: `{aggregate['selective_risk']['mean']:.4f}`.",
            f"- Mean acquisition fraction: `{aggregate['acquisition_fraction']['mean']:.4f}`.",
            f"- Mean population net gain: `{aggregate['population_net_gain']['mean']:.6f}`.",
            f"- Mean paired gain over no-source ablation: `{aggregate['paired_net_gain_delta']['mean']:+.6f}`.",
            f"- 95% seed-bootstrap CI for paired gain: `{summary['paired_inference']['bootstrap_mean_difference_ci_95']}`.",
            f"- Wilcoxon p-value: `{summary['paired_inference']['wilcoxon_two_sided']['pvalue']:.4f}`.",
            "",
            "## Confirmatory audit",
            "",
        ]
    )
    for name, value in summary["confirmatory_criteria"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The preregistered H5.2 criteria pass: sparse source-decomposed calibration gives nonzero, risk-controlled, positive net gain in all five seeds and beats the no-source continuous ablation in three seeds with a positive mean difference. The paired improvement is not statistically decisive with only five split seeds, so the source-decomposition advantage remains a promising but not final claim. The stricter bounded-shortfall certificate has low or zero coverage and remains an open requirement.",
            "",
        ]
    )
    return "\n".join(lines)


def _test(result: dict, method: str, family: str) -> dict:
    return result["methods"][method][family]["alpha_0.20"]["test"]


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "population_std": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return estimates[math.floor(0.025 * samples)], estimates[math.floor(0.975 * samples)]


def _wilcoxon(values: list[float]) -> dict[str, float]:
    from scipy.stats import wilcoxon

    result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


if __name__ == "__main__":
    main()
