from __future__ import annotations

import unittest

from harness_matsci.schema import ActionRecord
from harness_matsci.scientific_gain import GainTarget
from harness_matsci.simulator_views import CandidateSimulatorTool


def make_record(index: int, split: str) -> ActionRecord:
    value = index / 40.0
    worthy = int(index >= 32)
    return ActionRecord(
        record_id=f"r-{split}-{index}", benchmark="task", split=split,
        visible_context=f"candidate value {value}", candidate_action="evaluate candidate",
        action_type="choose_candidate", evidence=["base"],
        features={"x": value, "cost": 0.2, "reversibility": 0.8},
        label=worthy, utility=value,
    )


def target(record: ActionRecord) -> GainTarget:
    gain = record.features["x"] - 0.8
    return GainTarget(record.record_id, record.benchmark, record.utility, 0.2, 0.0, gain, 0.0, gain, int(gain > 0))


class SimulatorViewTests(unittest.TestCase):
    def test_candidate_specific_view_has_no_route(self) -> None:
        train = [make_record(index, "train") for index in range(40)]
        feedback = [make_record(index, "feedback") for index in range(20, 40)]
        tool = CandidateSimulatorTool(train, [target(row) for row in train], feedback, [target(row) for row in feedback])
        view = tool.evidence_view(make_record(39, "test"))
        self.assertEqual(view.view_id, "tool::candidate_simulator")
        self.assertIn("candidate_simulator_probability", view.record.features)
        self.assertNotIn("route", view.provenance)
        self.assertTrue(view.provenance["target_outcome_excluded"])


if __name__ == "__main__":
    unittest.main()
