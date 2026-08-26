from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.risk_control import hoeffding_upper_confidence


PRIMARY = "learned_portfolio_kg_mean"
BASELINES = (
    "base_factorized_lcb",
    "fixed_screening_lcb",
    "fixed_full_evidence_lcb",
    "static_reliability",
    "shadow_price_successive_halving",
)


def audit_acceptance(
    payload: dict[str, object],
    *,
    llm_gate_complete: bool = False,
    strongest_llm_trajectory_gain: float | None = None,
    strongest_llm_selective_risk: float | None = None,
) -> dict[str, object]:
    acceptance = payload["summary"]["acceptance"]
    primary = acceptance[PRIMARY]
    selected_count = int(primary["selected_count"])
    errors = int(primary["errors"])
    outcome_blind_hoeffding_upper = hoeffding_upper_confidence(
        [1.0] * errors + [0.0] * (selected_count - errors),
        0.05,
    )
    baseline_gains = {
        name: float(acceptance[name]["trajectory_total_strict_gain"])
        for name in BASELINES
    }
    llm_passed = False
    if llm_gate_complete:
        if strongest_llm_trajectory_gain is None or strongest_llm_selective_risk is None:
            raise ValueError("complete LLM gate requires gain and risk")
        llm_passed = (
            float(primary["trajectory_total_strict_gain"])
            > float(strongest_llm_trajectory_gain)
            and float(primary["selective_risk"])
            <= float(strongest_llm_selective_risk)
        )
    checks = {
        "selected_nonempty": int(primary["selected_count"]) > 0,
        "risk_ucb_le_0.10": float(primary["risk_upper_bound"]) <= 0.10,
        "outcome_blind_hoeffding_ucb_le_0.10": (
            outcome_blind_hoeffding_upper <= 0.10
        ),
        "positive_trajectory_gain": (
            float(primary["trajectory_total_strict_gain"]) > 0.0
        ),
        "beats_every_non_llm_baseline": all(
            float(primary["trajectory_total_strict_gain"]) > gain
            for gain in baseline_gains.values()
        ),
        "llm_same_evidence_gate_complete": bool(llm_gate_complete),
        "beats_strongest_same_evidence_llm": bool(llm_passed),
    }
    non_llm_passed = all(
        checks[name]
        for name in (
            "selected_nonempty",
            "risk_ucb_le_0.10",
            "outcome_blind_hoeffding_ucb_le_0.10",
            "positive_trajectory_gain",
            "beats_every_non_llm_baseline",
        )
    )
    full_passed = non_llm_passed and llm_gate_complete and llm_passed
    return {
        "schema_version": "h5-round2-acceptance-audit-v2",
        "primary_method": PRIMARY,
        "checks": checks,
        "non_llm_acceptance_gate_passed": non_llm_passed,
        "full_acceptance_gate_passed": full_passed,
        "test_unlocked": full_passed,
        "primary": primary,
        "outcome_blind_hoeffding_risk_upper_bound": outcome_blind_hoeffding_upper,
        "outcome_blind_hoeffding_scope": (
            "conditional selected-batch average risk under outcome-blind selection, "
            "independent bounded losses, and a harness frozen before evaluation"
        ),
        "baseline_trajectory_gains": baseline_gains,
        "strongest_same_evidence_llm": (
            {
                "trajectory_total_strict_gain": strongest_llm_trajectory_gain,
                "selective_risk": strongest_llm_selective_risk,
            }
            if llm_gate_complete
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-gate-complete", action="store_true")
    parser.add_argument("--strongest-llm-trajectory-gain", type=float)
    parser.add_argument("--strongest-llm-selective-risk", type=float)
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    audit = audit_acceptance(
        payload,
        llm_gate_complete=args.llm_gate_complete,
        strongest_llm_trajectory_gain=args.strongest_llm_trajectory_gain,
        strongest_llm_selective_risk=args.strongest_llm_selective_risk,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
