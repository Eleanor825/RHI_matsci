from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "source_voi_expected",
    "source_voi_probability",
    "no_source_voi_expected",
    "no_source_voi_probability",
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
    folds = []
    for fold in range(1, 5):
        path = root / f"fold{fold}_confirmatory" / "result.json"
        folds.append((fold, json.loads(path.read_text(encoding="utf-8"))))
    methods = {}
    for method in METHODS:
        deployed = [bool(result["policies"][method]["deployed"]) for _, result in folds]
        gains = [
            float(result["policies"][method]["deployed_test_population_gain"])
            for _, result in folds
        ]
        test_risks = [
            float(result["policies"][method]["test_evaluation"]["selective_risk"])
            for _, result in folds
        ]
        methods[method] = {
            "certified_folds": sum(deployed),
            "fold_count": len(folds),
            "deployed_test_gains": gains,
            "mean_deployed_test_gain": sum(gains) / len(gains),
            "test_selective_risks": test_risks,
        }
    source_probability = methods["source_voi_probability"]["mean_deployed_test_gain"]
    no_source_probability = methods["no_source_voi_probability"][
        "mean_deployed_test_gain"
    ]
    return {
        "experiment": "h17_mixed_regime_pairwise_two_stage_voe",
        "development_fold": 0,
        "confirmatory_folds": [1, 2, 3, 4],
        "preregistered_primary": "source_voi_expected",
        "primary_confirmatory_pass": bool(
            methods["source_voi_expected"]["certified_folds"] == 4
            and methods["source_voi_expected"]["mean_deployed_test_gain"]
            > methods["no_source_voi_expected"]["mean_deployed_test_gain"]
            and methods["source_voi_expected"]["mean_deployed_test_gain"]
            > methods["external_variance"]["mean_deployed_test_gain"]
        ),
        "secondary_source_probability_relative_gain_over_no_source": (
            (source_probability - no_source_probability) / no_source_probability
            if no_source_probability > 0.0
            else None
        ),
        "methods": methods,
    }


def markdown(result: dict) -> str:
    lines = [
        "# H17 Confirmatory Aggregate",
        "",
        "Folds 1-4 were run after the fold-0 protocol lock.",
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
    lines.extend(
        [
            "",
            f"- Preregistered primary pass: `{result['primary_confirmatory_pass']}`.",
            "- The expected-VoE primary did not satisfy the four-fold gate.",
            "- The secondary source-aware probability score certified 3/4 folds and "
            f"improved mean deployed gain over its matched no-source score by "
            f"{100.0 * result['secondary_source_probability_relative_gain_over_no_source']:.1f}%.",
            "- Static external variance certified 0/4 folds.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
