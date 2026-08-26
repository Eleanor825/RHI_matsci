from __future__ import annotations

import unittest

from harness_matsci.matxbot_runtime import (
    MatXBotOutcomeEvent,
    MatXBotPreActionEvent,
    pair_matxbot_events,
)


def _pre_action(**updates) -> MatXBotPreActionEvent:
    values = {
        "event_id": "run-1::step-1",
        "run_id": "run-1",
        "parent_step_id": None,
        "system": "lfp",
        "stage": "screening",
        "visible_context": {"candidate": "electrolyte-A"},
        "proposed_action": {"action": "simulate"},
        "available_tools": ("simulator",),
        "model_replica_readouts": ({"replica_id": "m1", "expected_gain": 0.1},),
        "evidence_replica_outputs": ({"view_id": "e1", "margin": 0.2},),
        "estimated_cost": 0.1,
        "reversibility": 0.9,
        "timestamp": "2026-08-22T00:00:00Z",
    }
    values.update(updates)
    return MatXBotPreActionEvent(**values)


class MatXBotRuntimeTests(unittest.TestCase):
    def test_pre_action_rejects_hidden_outcome_keys(self) -> None:
        with self.assertRaises(ValueError):
            _pre_action(visible_context={"candidate": "A", "scientific_utility": 0.8})

    def test_pairs_delayed_outcome_without_copying_it_into_pre_action(self) -> None:
        pre = _pre_action()
        outcome = MatXBotOutcomeEvent(
            event_id="outcome-1",
            pre_action_event_id=pre.event_id,
            tool_results=({"tool": "simulator", "result": "stable"},),
            execution_cost=0.1,
            observed_success=1,
            scientific_utility=0.8,
            failure_harm=0.0,
            next_stage="selection",
            timestamp="2026-08-22T00:01:00Z",
        )
        paired = pair_matxbot_events([pre], [outcome])[0]
        self.assertTrue(paired["outcome_available"])
        self.assertNotIn("scientific_utility", paired["pre_action"]["visible_context"])
        self.assertEqual(
            paired["outcome_for_training_and_evaluation_only"]["scientific_utility"],
            0.8,
        )

    def test_rejects_unknown_outcome_reference(self) -> None:
        outcome = MatXBotOutcomeEvent(
            event_id="outcome-1",
            pre_action_event_id="missing",
            tool_results=(),
            execution_cost=0.0,
            observed_success=0,
            scientific_utility=0.0,
            failure_harm=0.2,
            next_stage=None,
            timestamp="2026-08-22T00:01:00Z",
        )
        with self.assertRaises(ValueError):
            pair_matxbot_events([_pre_action()], [outcome])


if __name__ == "__main__":
    unittest.main()
