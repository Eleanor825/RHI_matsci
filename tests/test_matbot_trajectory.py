import unittest

from harness_matsci.matbot_trajectory import (
    assemble_matbot_trajectories,
    trajectory_summary,
)


def pre_action(step=0, capture_mode="live_runtime"):
    return {
        "schema_version": "matxbot-uncertainty-v1",
        "event_type": "pre_action",
        "event_id": f"pre-{step}",
        "run_id": "run-1",
        "trajectory_id": "trajectory-1",
        "step_id": step,
        "timestamp": "2026-08-22T12:00:00Z",
        "capture_mode": capture_mode,
        "system": "lfp",
        "stage": "screening",
        "action_id": "simulate",
        "action_type": "tool_call",
        "visible_context": "visible",
        "evidence": ["source"],
        "internal_signals": {"sample_variance": 0.1},
        "external_signals": {"source_disagreement": 0.2},
        "cost_signals": {"normalized_cost": 0.05},
        "uncertainty_output": {
            "p_success": 0.7,
            "expected_net_scientific_gain": 0.2,
            "internal_variance": 0.1,
            "external_variance": 0.2,
            "expected_value_of_evidence": 0.1,
            "positive_gain_probability": 0.8,
        },
        "hidden_outcome_exposed": False,
    }


def post_outcome(step=0, terminal=True, capture_mode="live_runtime"):
    return {
        "schema_version": "matxbot-uncertainty-v1",
        "event_type": "post_outcome",
        "event_id": f"post-{step}",
        "parent_event_id": f"pre-{step}",
        "run_id": "run-1",
        "trajectory_id": "trajectory-1",
        "step_id": step,
        "timestamp": "2026-08-22T12:01:00Z",
        "capture_mode": capture_mode,
        "outcome_status": "success",
        "observed_outcome": {"scientific_utility": 0.4},
        "realized_cost": 0.05,
        "terminal": terminal,
    }


class MatBotTrajectoryTests(unittest.TestCase):
    def test_assembles_complete_live_trajectory(self):
        trajectories = assemble_matbot_trajectories([pre_action(), post_outcome()])
        self.assertEqual(len(trajectories), 1)
        self.assertTrue(trajectories[0].complete)
        self.assertTrue(trajectories[0].is_live_runtime)
        self.assertEqual(trajectory_summary(trajectories)["live_runtime_count"], 1)

    def test_rejects_selected_route(self):
        event = pre_action()
        event["uncertainty_output"]["selected_route"] = 1
        with self.assertRaises(ValueError):
            assemble_matbot_trajectories([event])

    def test_rejects_hidden_outcome_in_pre_action(self):
        event = pre_action()
        event["hidden_outcome_exposed"] = True
        with self.assertRaises(ValueError):
            assemble_matbot_trajectories([event])


if __name__ == "__main__":
    unittest.main()
