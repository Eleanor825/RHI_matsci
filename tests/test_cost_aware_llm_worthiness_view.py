import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "sd_worth"
    / "run_sequential_llm_worthiness.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("llm_worthiness_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _trajectory():
    def row(stage, cost, evidence):
        return {
            "stage": stage,
            "tool_cost": cost,
            "visible_observation": {
                "record": {
                    "record_id": "r1",
                    "benchmark": "task",
                    "split": "test",
                    "action_type": "choose_candidate",
                    "candidate_action": "Execute A.",
                    "visible_context": "Visible context.",
                    "evidence": evidence,
                    "features": {"cost": 0.2, "reversibility": 0.8},
                    "metadata": {},
                },
                "provenance": {"hidden_outcome_exposed": False},
            },
        }

    return {
        "record_id": "r1",
        "stages": [
            row("base", 0.0, ["base"]),
            row("screening", 0.025, ["base", "screen"]),
            row("verification", 0.075, ["base", "verify"]),
        ],
    }


def test_cost_aware_bundle_merges_evidence_and_charges_both_tools():
    trajectory = _trajectory()

    view = MODULE._view(
        trajectory,
        "screening+verification",
        cost_aware=True,
    )

    assert view.view_id == "sequential-v2-cost-aware::screening+verification"
    assert view.record.evidence == ["base", "screen", "verify"]
    assert view.record.features["cost"] == pytest.approx(0.3)
    assert MODULE._acquired_cost(
        trajectory, "screening+verification"
    ) == pytest.approx(0.1)
