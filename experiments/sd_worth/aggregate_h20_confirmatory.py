from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "source_multitool_expected",
    "no_source_multitool_expected",
    "fixed_high_fidelity_expected",
    "static_reliability_high_fidelity",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(Path(args.results_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")


def aggregate(root: Path) -> dict:
    folds = [
        (fold, json.loads((root / f"fold{fold}" / "result.json").read_text()))
        for fold in range(5)
    ]
    methods = {}
    for method in METHODS:
        rows = [result["policies"][method] for _, result in folds]
        gains = [float(row["deployed_test_population_gain"]) for row in rows]
        selected = [tuple(row["selected_tools"]) for row in rows]
        test_risks = [
            float(row["test_evaluation"]["selective_risk"])
            if row["test_evaluation"] is not None
            else 0.0
            for row in rows
        ]
        methods[method] = {
            "certified_folds": sum(bool(row["deployed"]) for row in rows),
            "fold_count": len(rows),
            "deployed_test_gains": gains,
            "mean_deployed_test_gain": sum(gains) / len(gains),
            "selected_tool_subsets": [list(value) for value in selected],
            "non_high_fidelity_folds": sum(
                any(tool != "high_fidelity" for tool in subset)
                for subset in selected
            ),
            "test_selective_risks": test_risks,
        }
    primary = methods["source_multitool_expected"]
    criteria = {
        "deploys_at_least_four_of_five": primary["certified_folds"] >= 4,
        "positive_mean_gain": primary["mean_deployed_test_gain"] > 0.0,
        "beats_no_source": (
            primary["mean_deployed_test_gain"]
            > methods["no_source_multitool_expected"]["mean_deployed_test_gain"]
        ),
        "beats_fixed_high_fidelity": (
            primary["mean_deployed_test_gain"]
            > methods["fixed_high_fidelity_expected"]["mean_deployed_test_gain"]
        ),
        "uses_non_high_fidelity_on_two_folds": primary["non_high_fidelity_folds"] >= 2,
        "observed_test_risk_within_alpha": all(
            risk <= 0.20 for risk in primary["test_selective_risks"]
        ),
    }
    return {
        "experiment": "h20_perovskite_multitool_external_validation",
        "protocol_hash": "6e29eb36b0065749fcdd0c7f908875750a5a7a56470557bbe9b92e9cd7f91c03",
        "confirmatory_folds": [0, 1, 2, 3, 4],
        "preregistered_primary": "source_multitool_expected",
        "success_criteria": criteria,
        "primary_confirmatory_pass": all(criteria.values()),
        "methods": methods,
    }


def markdown(result):
    lines = [
        "# H20 Perovskite Multi-Tool Confirmatory Aggregate",
        "",
        "| method | certified folds | mean test gain | non-HF folds | selected subsets |",
        "|---|---:|---:|---:|---|",
    ]
    for method in METHODS:
        row = result["methods"][method]
        subsets = ", ".join(
            "+".join(values) if values else "none"
            for values in row["selected_tool_subsets"]
        )
        lines.append(
            f"| `{method}` | {row['certified_folds']}/{row['fold_count']} | "
            f"{row['mean_deployed_test_gain']:+.6f} | "
            f"{row['non_high_fidelity_folds']} | {subsets} |"
        )
    lines.extend(["", "## Frozen criteria", ""])
    for name, passed in result["success_criteria"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            "",
            f"Overall H20 confirmatory pass: `{result['primary_confirmatory_pass']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
