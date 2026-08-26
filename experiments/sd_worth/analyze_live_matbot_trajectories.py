from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.live_trajectory_analysis import analyze_live_trajectories
from harness_matsci.matbot_trajectory import load_matbot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    trajectories = []
    for path in args.trajectory:
        trajectories.extend(load_matbot_trajectories(path))
    analysis = analyze_live_trajectories(trajectories)
    payload = analysis.to_json()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    Path(args.output_md).write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


def _render_markdown(payload: dict[str, object]) -> str:
    numeric = payload["numeric_outputs"]
    rows = []
    for field, summary in numeric.items():
        values = ", ".join(f"{value:.4f}" for value in summary["values"])
        rows.append(
            f"| `{field}` | {values} | {summary['mean']:.4f} | "
            f"{summary['range']:.4f} | {summary['population_variance']:.6f} |"
        )
    return "\n".join(
        [
            "# Live MatBot Cross-Model Analysis",
            "",
            "## Scope",
            "",
            f"- Live trajectories: {payload['trajectory_count']}",
            f"- Distinct model backends: {payload['model_count']}",
            f"- Trajectories with observed terminal outcomes: {payload['complete_count']}",
            f"- Tool calls/errors: {payload['total_tool_calls']}/{payload['total_tool_errors']} "
            f"(error rate {payload['tool_error_rate']:.3f})",
            "",
            "## Numeric Outputs",
            "",
            "| Field | Per-model values | Mean | Range | Population variance |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Action Disagreement",
            "",
            f"- Pair count: {payload['action_similarity']['pair_count']}",
            "- Mean/min character-bigram Jaccard: "
            f"{payload['action_similarity']['character_bigram_mean_jaccard']:.4f} / "
            f"{payload['action_similarity']['character_bigram_min_jaccard']:.4f}",
            "- Mean/min token Jaccard: "
            f"{payload['action_similarity']['token_mean_jaccard']:.4f} / "
            f"{payload['action_similarity']['token_min_jaccard']:.4f}",
            "",
            "## Within-Model Repeated Sampling",
            "",
            _render_within_model(
                payload["within_model_numeric_dispersion"],
                payload["within_model_action_similarity"],
            ),
            "",
            "## Valid Interpretation",
            "",
            "- The dispersion above is **between different model backends**. It is not "
            "within-model internal variance.",
            "- The models' `reported_internal_uncertainty` and "
            "`reported_external_uncertainty` fields remain "
            "uncalibrated self-reports until repeated sampling, controlled evidence "
            "interventions, and observed outcomes are available.",
            "- These trajectories validate real runtime capture and tool traces, but not "
            f"scientific benefit: all {payload['trajectory_count']} trajectories lack a "
            "post-outcome event.",
            "",
        ]
    )


def _render_within_model(
    groups: dict[str, object],
    action_groups: dict[str, object],
) -> str:
    if not groups:
        return "No model has two successful independent trajectories."
    lines = []
    for model, fields in groups.items():
        lines.append(f"- `{model}` ({len(next(iter(fields.values()))['values'])} samples):")
        action = action_groups[model]
        lines.append(
            f"  - action token Jaccard={action['token_mean_jaccard']:.4f}; "
            f"character-bigram Jaccard={action['character_bigram_mean_jaccard']:.4f}"
        )
        lines.extend(
            f"  - `{field}` range={summary['range']:.4f}, "
            f"population variance={summary['population_variance']:.6f}"
            for field, summary in fields.items()
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
