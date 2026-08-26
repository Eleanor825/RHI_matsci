import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "sd_worth"
    / "build_adaptive_evidence_trajectories.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("adaptive_builder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Observation:
    tool_cost = 0.025

    def __init__(self, stage, replica):
        self.stage = stage
        self.replica = replica

    def visible_payload(self):
        return {
            "record": {
                "visible_context": "visible",
                "candidate_action": "act",
                "evidence": [f"{self.stage}-{self.replica}"],
                "features": {},
                "metadata": {},
            },
            "provenance": {"hidden_outcome_exposed": False},
        }


class Environment:
    def observe(self, record, *, stage, evidence_replica, run_seed):
        return Observation(stage, evidence_replica)


def test_prior_views_are_replica_crossed_and_label_blind(monkeypatch):
    monkeypatch.setattr(MODULE, "audit_visible_payload", lambda payload: None)

    rows = MODULE._prior_views(Environment(), object(), replicas=3, run_seed=7)

    assert len(rows) == 6
    assert {(row["stage"], row["evidence_replica"]) for row in rows} == {
        (stage, replica)
        for stage in ("screening", "verification")
        for replica in range(3)
    }
    assert all(
        row["visible_observation"]["provenance"]["hidden_outcome_exposed"] is False
        for row in rows
    )


def test_adaptive_fold_assignment_is_deterministic():
    assert MODULE._fold(7, "task::group", 5) == MODULE._fold(
        7, "task::group", 5
    )


def test_structured_environment_is_the_default_non_degenerate_prior():
    assert MODULE._environment_class("structured") is MODULE.TrainOnlyStructuredSurrogateToolEnvironment
    assert MODULE._environment_class("real") is MODULE.RealScientificToolEnvironment
