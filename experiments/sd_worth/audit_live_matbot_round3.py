from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness_matsci.matbot_trajectory import load_matbot_trajectories


FROZEN_HASHES = {
    "src/harness_matsci/real_scientific_tools.py": "5138afd84cdbee7c195befc3c6b61f49e1b7c9396452a4cece7f2cc3f2667561",
    "experiments/sd_worth/capture_live_matbot_tool_trajectories.py": "5597779cf8a35cdc0c40a07ed60695d13f5cf936762f472c9a74631f29c1ba9a",
    "experiments/sd_worth/evaluate_learned_live_matbot_harness.py": "d1b4d65df2d365ce95701d7df1ff0b6e91b638e2707b3bc10d34fd93cb538349",
    "src/harness_matsci/matbot_strict_gain_runtime.py": "a60949fd6c28ead9fe393a17d0b64d71ff59aafd11294ead6e7f33b4afaf9d73",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--export-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    export = json.loads(Path(args.export_manifest).read_text(encoding="utf-8"))
    trajectories = load_matbot_trajectories(args.trajectories)
    primary = results["learned_global"]
    baseline_gains = {
        name: row["trajectory_total_strict_gain"] for name, row in results["baselines"].items()
    }
    baseline_risks = {name: row["selective_risk"] for name, row in results["baselines"].items()}
    forbidden_routes = sum(
        "selected_route" in step.uncertainty_output for trajectory in trajectories for step in trajectory.steps
    )
    hidden_pre_actions = sum(
        any(
            key in json.dumps(
                {
                    "visible_context": step.visible_context,
                    "evidence": step.evidence,
                    "internal_signals": step.internal_signals,
                    "external_signals": step.external_signals,
                    "tool_trace": step.tool_trace,
                },
                sort_keys=True,
            ).lower()
            for key in ("hidden_outcome_for_evaluation_only", '"observed_success"', '"strict_realized_net_gain"')
        )
        for trajectory in trajectories
        for step in trajectory.steps
    )
    root = Path(args.repo_root)
    current_hashes = {path: _sha256(root / path) for path in FROZEN_HASHES}
    checks = {
        "fresh_fingerprint_overlap_zero": export["scientific_fingerprint_overlap"] == 0,
        "live_runtime_only": bool(trajectories) and all(row.is_live_runtime for row in trajectories),
        "all_trajectories_complete": bool(trajectories) and all(row.complete for row in trajectories),
        "numeric_output_no_route": forbidden_routes == 0,
        "hidden_outcome_absent_pre_action": hidden_pre_actions == 0,
        "frozen_hashes_match": current_hashes == FROZEN_HASHES,
        "selected_nonempty": primary["selected"] > 0,
        "risk_ucb_le_0.10": primary["risk_upper_bound_95"] <= 0.10,
        "positive_trajectory_gain": primary["trajectory_total_strict_gain"] > 0.0,
        "beats_every_baseline": all(
            primary["trajectory_total_strict_gain"] > value for value in baseline_gains.values()
        ),
        "no_higher_empirical_risk_than_best_gain_baseline": primary["selective_risk"]
        <= baseline_risks[max(baseline_gains, key=baseline_gains.get)],
    }
    report = {
        "schema_version": "live-matbot-round3-audit-v1",
        "checks": checks,
        "gate_passed": all(checks.values()),
        "trajectory_count": len(trajectories),
        "primary": primary,
        "baseline_gains": baseline_gains,
        "baseline_risks": baseline_risks,
        "forbidden_route_count": forbidden_routes,
        "hidden_pre_action_count": hidden_pre_actions,
        "frozen_hashes": FROZEN_HASHES,
        "current_hashes": current_hashes,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
