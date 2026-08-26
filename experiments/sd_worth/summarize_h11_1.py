from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY = "task_sparse_tool_margin_continuous_sd_worth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h10-dir", required=True)
    parser.add_argument("--rerun-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    h10_dir = Path(args.h10_dir)
    rerun_dir = Path(args.rerun_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(h10_dir.glob("seed_*.json"), key=lambda item: int(item.stem.split("_")[-1])):
        source = json.loads(path.read_text(encoding="utf-8"))
        seed = int(source["seed"])
        rerun_path = rerun_dir / path.name
        if rerun_path.exists():
            rerun = json.loads(rerun_path.read_text(encoding="utf-8"))
            activation = rerun["safe_activation"]["source_decomposed"]
            provenance = "direct_h11_1_rerun"
        else:
            selection = source["methods"][PRIMARY]["risk_control"]["alpha_0.20"]["acceptance"]
            if int(selection["selected_count"]) != 0:
                raise ValueError(f"seed {seed} requires an H11.1 rerun")
            activation = {
                "selected_policy": "no_op",
                "risk_selection": selection,
                "acceptance": None,
                "test": _no_op_test(),
            }
            provenance = "deterministic_zero_certificate_replay"
        test = activation["test"]
        rows.append(
            {
                "seed": seed,
                "selected_policy": activation["selected_policy"],
                "provenance": provenance,
                "coverage": float(test["coverage"]),
                "selective_risk": float(test["selective_risk"]),
                "population_net_gain": float(test["population_net_gain"]),
                "selected_count": int(test["selected_count"]),
            }
        )

    summary = {
        "experiment": "h11_1_safe_activation_five_seeds",
        "rows": rows,
        "active_seeds": sum(row["selected_policy"] != "no_op" for row in rows),
        "nonnegative_gain_seeds": sum(row["population_net_gain"] >= 0.0 for row in rows),
        "positive_gain_seeds": sum(row["population_net_gain"] > 0.0 for row in rows),
        "nonzero_coverage_seeds": sum(row["coverage"] > 0.0 for row in rows),
        "risk_at_most_0.20_seeds": sum(row["selective_risk"] <= 0.20 for row in rows),
        "mean_population_net_gain": sum(row["population_net_gain"] for row in rows) / len(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(_markdown(summary), encoding="utf-8")


def _no_op_test() -> dict[str, float | int]:
    return {
        "coverage": 0.0,
        "selective_risk": 0.0,
        "population_net_gain": 0.0,
        "selected_count": 0,
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# H11.1 Five-Seed Safe-Activation Replay",
        "",
        "| seed | policy | provenance | coverage | risk | population net gain |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['seed']} | `{row['selected_policy']}` | `{row['provenance']}` | "
            f"{row['coverage']:.4f} | {row['selective_risk']:.4f} | "
            f"{row['population_net_gain']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Active seeds: `{summary['active_seeds']}/5`.",
            f"- Nonnegative-gain seeds: `{summary['nonnegative_gain_seeds']}/5`.",
            f"- Positive-gain seeds: `{summary['positive_gain_seeds']}/5`.",
            f"- Nonzero-coverage seeds: `{summary['nonzero_coverage_seeds']}/5`.",
            f"- Risk at most 0.20: `{summary['risk_at_most_0.20_seeds']}/5`.",
            f"- Mean population net gain: `{summary['mean_population_net_gain']:+.6f}`.",
            "",
            "H11.1 removes negative tool cost in zero-certificate seeds but does not solve "
            "the underlying generalization problem: only seed 7 activates and obtains positive gain.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
