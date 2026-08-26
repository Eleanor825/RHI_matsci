from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "source_voi_expected",
    "source_voi_probability",
    "no_source_voi_expected",
    "no_source_voi_probability",
    "source_voi_tail",
    "no_source_voi_tail",
    "external_variance",
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
        deployed = [bool(result["policies"][method]["deployed"]) for _, result in folds]
        gains = [
            float(result["policies"][method]["deployed_test_population_gain"])
            for _, result in folds
        ]
        risks = [
            float(result["policies"][method]["test_evaluation"]["selective_risk"])
            for _, result in folds
        ]
        methods[method] = {
            "certified_folds": sum(deployed),
            "fold_count": len(folds),
            "deployed_test_gains": gains,
            "mean_deployed_test_gain": sum(gains) / len(gains),
            "test_selective_risks": risks,
        }
    primary = methods["source_voi_expected"]
    no_source = methods["no_source_voi_expected"]
    static = methods["external_variance"]
    criteria = {
        "certifies_at_least_four_of_five": primary["certified_folds"] >= 4,
        "positive_mean_gain": primary["mean_deployed_test_gain"] > 0.0,
        "beats_no_source_expected": (
            primary["mean_deployed_test_gain"] > no_source["mean_deployed_test_gain"]
        ),
        "beats_static_external_variance": (
            primary["mean_deployed_test_gain"] > static["mean_deployed_test_gain"]
        ),
    }
    return {
        "experiment": "h19_phonons_frozen_external_validation",
        "confirmatory_folds": [0, 1, 2, 3, 4],
        "protocol_hash": "1404b7c7163b7f8480c16a330e4de352ab638ce59bb9bdce2416f554c7cf4ef6",
        "preregistered_primary": "source_voi_expected",
        "success_criteria": criteria,
        "primary_confirmatory_pass": all(criteria.values()),
        "methods": methods,
    }


def markdown(result: dict) -> str:
    lines = [
        "# H19 Frozen External Validation Aggregate",
        "",
        "All five folds were run after the protocol lock and none were used for method selection.",
        "",
        "| method | certified folds | mean deployed test gain | fold gains |",
        "|---|---:|---:|---|",
    ]
    for method in METHODS:
        row = result["methods"][method]
        gains = ", ".join(f"{value:+.6f}" for value in row["deployed_test_gains"])
        lines.append(
            f"| `{method}` | {row['certified_folds']}/{row['fold_count']} | "
            f"{row['mean_deployed_test_gain']:+.6f} | {gains} |"
        )
    lines.extend(["", "## Frozen success criteria", ""])
    for criterion, passed in result["success_criteria"].items():
        lines.append(f"- `{criterion}`: `{passed}`")
    lines.extend(
        [
            "",
            f"Overall H19 confirmatory pass: `{result['primary_confirmatory_pass']}`.",
            "",
            "The primary produced positive risk-controlled gain and beat static variance, "
            "but it certified on only three folds and did not beat the matched no-source estimator. "
            "H19 therefore falsifies the current claim that external-source decomposition reliably "
            "improves aggregate acquisition utility on a new property task.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
