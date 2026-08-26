from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev


PRIMARY = "task_sparse_continuous_sd_worth"


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
    fixed_rows = []
    evolved_rows = []
    bernstein_rows = []
    for result in results:
        fixed = result["methods"][PRIMARY]["hoeffding_shortfall_risk_control"][
            "alpha_0.20"
        ]
        fixed_rows.append(_row(result["seed"], fixed["test"], fixed["acceptance"]))
        bernstein = result["methods"][PRIMARY]["shortfall_risk_control"]["alpha_0.20"]
        bernstein_rows.append(_row(result["seed"], bernstein["test"], bernstein["acceptance"]))
        evolution = result["harness_evolution_hoeffding_shortfall"]
        evolved_rows.append(
            {
                **_row(
                    result["seed"],
                    evolution["selected_result"]["test"],
                    evolution["selected_result"]["risk_selection"],
                ),
                "selected_harness": evolution["selected_harness"],
                "accepted_mutations": [
                    item["candidate"]
                    for item in evolution["history"][1:]
                    if item["accepted"]
                ],
            }
        )
    criteria = {
        "fixed_nonzero_coverage_seeds": _count(fixed_rows, "coverage", lambda x: x > 0.0),
        "fixed_positive_selected_gain_seeds": _count(
            fixed_rows, "selected_mean_gain", lambda x: x > 0.0
        ),
        "fixed_positive_population_net_gain_seeds": _count(
            fixed_rows, "population_net_gain", lambda x: x > 0.0
        ),
        "fixed_test_shortfall_at_most_0.20_seeds": _count(
            fixed_rows, "selected_mean_shortfall", lambda x: x <= 0.20
        ),
        "evolved_nonzero_coverage_seeds": _count(
            evolved_rows, "coverage", lambda x: x > 0.0
        ),
        "evolved_positive_population_net_gain_seeds": _count(
            evolved_rows, "population_net_gain", lambda x: x > 0.0
        ),
        "evolved_test_shortfall_at_most_0.20_seeds": _count(
            evolved_rows, "selected_mean_shortfall", lambda x: x <= 0.20
        ),
        "bernstein_nonzero_coverage_seeds": _count(
            bernstein_rows, "coverage", lambda x: x > 0.0
        ),
    }
    criteria["h8_fixed_confirmatory_pass"] = (
        criteria["fixed_nonzero_coverage_seeds"] >= 4
        and criteria["fixed_positive_selected_gain_seeds"] >= 4
        and criteria["fixed_positive_population_net_gain_seeds"] >= 4
    )
    criteria["h8_evolved_supportive_pass"] = (
        criteria["evolved_nonzero_coverage_seeds"] >= 4
        and criteria["evolved_positive_population_net_gain_seeds"] >= 4
        and criteria["evolved_test_shortfall_at_most_0.20_seeds"] >= 4
    )
    return {
        "experiment": "h8_hoeffding_certified_utility_shortfall",
        "alpha": 0.20,
        "primary_method": PRIMARY,
        "fixed_rows": fixed_rows,
        "evolved_rows": evolved_rows,
        "empirical_bernstein_rows": bernstein_rows,
        "fixed_aggregate": _aggregate_rows(fixed_rows),
        "evolved_aggregate": _aggregate_rows(evolved_rows),
        "confirmatory_criteria": criteria,
    }


def _row(seed: int, test: dict, certificate: dict) -> dict:
    return {
        "seed": seed,
        "coverage": test["coverage"],
        "selected_mean_gain": test["selected_mean_gain"],
        "population_net_gain": test["population_net_gain"],
        "selected_mean_shortfall": test["selected_mean_shortfall"],
        "binary_selective_risk": test["selective_risk"],
        "acquisition_fraction": test["acquisition_fraction"],
        "acceptance_empirical_shortfall": certificate["empirical_risk"],
        "acceptance_shortfall_ucb": certificate["risk_upper_bound"],
    }


def _aggregate_rows(rows: list[dict]) -> dict:
    return {
        metric: _aggregate([row[metric] for row in rows])
        for metric in (
            "coverage",
            "selected_mean_gain",
            "population_net_gain",
            "selected_mean_shortfall",
            "binary_selective_risk",
            "acquisition_fraction",
        )
    }


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "population_std": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _count(rows: list[dict], key: str, predicate) -> int:
    return sum(predicate(row[key]) for row in rows)


def render(summary: dict) -> str:
    lines = [
        "# H8 Hoeffding-Certified Utility Shortfall",
        "",
        "The certificate controls normalized continuous utility shortfall, not binary failure probability.",
        "",
        "## Fixed source-decomposed harness",
        "",
        "| seed | coverage | selected gain | population net gain | mean shortfall | binary risk | acceptance UCB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fixed_rows"]:
        lines.append(_render_row(row))
    lines.extend(
        [
            "",
            "## Hoeffding-certified recursive harness evolution",
            "",
            "| seed | selected harness | accepted mutations | coverage | selected gain | population net gain | mean shortfall | binary risk | acceptance UCB |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["evolved_rows"]:
        lines.append(
            f"| {row['seed']} | `{row['selected_harness']}` | "
            f"`{','.join(row['accepted_mutations'])}` | {row['coverage']:.4f} | "
            f"{row['selected_mean_gain']:.4f} | {row['population_net_gain']:.6f} | "
            f"{row['selected_mean_shortfall']:.4f} | {row['binary_selective_risk']:.4f} | "
            f"{row['acceptance_shortfall_ucb']:.4f} |"
        )
    lines.extend(["", "## Confirmatory audit", ""])
    for key, value in summary["confirmatory_criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Hoeffding certification is evaluated on the same bounded shortfall loss, score grid, data splits, and gain targets as the earlier empirical-Bernstein certificate. Any coverage improvement therefore comes from concentration-bound power rather than a changed target. Test-set shortfall is a held-out empirical stress test; the formal statement requires acceptance/deployment exchangeability.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_row(row: dict) -> str:
    return (
        f"| {row['seed']} | {row['coverage']:.4f} | {row['selected_mean_gain']:.4f} | "
        f"{row['population_net_gain']:.6f} | {row['selected_mean_shortfall']:.4f} | "
        f"{row['binary_selective_risk']:.4f} | {row['acceptance_shortfall_ucb']:.4f} |"
    )


if __name__ == "__main__":
    main()
