import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.live_trajectory_analysis import analyze_live_trajectories
from harness_matsci.matbot_trajectory import (
    assemble_matbot_trajectories,
    load_matbot_trajectories,
)


def _event(event_id, trajectory_id, model, p_success, action, tool_error=False):
    return {
        "schema_version": "matxbot-uncertainty-v1",
        "capture_mode": "live_runtime",
        "event_type": "pre_action",
        "event_id": event_id,
        "trajectory_id": trajectory_id,
        "run_id": trajectory_id,
        "step_id": 0,
        "system": "ncm",
        "stage": "generation",
        "timestamp": "2026-08-22T12:00:00Z",
        "action_id": event_id,
        "action_type": "propose_experiment",
        "visible_context": "visible",
        "evidence": [],
        "internal_signals": {},
        "external_signals": {},
        "cost_signals": {},
        "uncertainty_output": {
            "p_success": p_success,
            "expected_scientific_utility": 0.8,
            "reported_internal_uncertainty": 0.2,
            "reported_external_uncertainty": 0.3,
            "expected_value_of_evidence": 0.4,
            "irreversible_risk": 0.1,
            "expected_net_scientific_gain": p_success * 0.8 - 0.2 - 0.1 * 0.5,
        },
        "agent_backend": model,
        "tool_trace": [{"is_error": tool_error}],
        "action_text": action,
        "rationale": "test",
        "hidden_outcome_exposed": False,
    }


class LiveTrajectoryAnalysisTests(unittest.TestCase):
    def test_separates_between_model_dispersion_from_internal_variance(self):
        trajectories = assemble_matbot_trajectories(
            [
                _event("a", "ta", "provider/model-a", 0.6, "test LiFSI", False),
                _event("b", "tb", "provider/model-b", 0.8, "test LiODFB", True),
            ]
        )
        result = analyze_live_trajectories(trajectories)
        self.assertEqual(result.model_count, 2)
        self.assertAlmostEqual(result.numeric_outputs["p_success"]["mean"], 0.7)
        self.assertAlmostEqual(result.numeric_outputs["p_success"]["population_variance"], 0.01)
        self.assertFalse(result.within_model_replication_available)
        self.assertFalse(result.reported_internal_uncertainty_is_calibrated)
        self.assertFalse(result.outcome_validation_available)
        self.assertEqual(result.total_tool_errors, 1)
        self.assertEqual(result.action_similarity["pair_count"], 1)

    def test_reports_same_model_repeated_sampling_separately(self):
        trajectories = assemble_matbot_trajectories(
            [
                _event("a", "ta", "provider/model-a", 0.6, "test LiFSI", False),
                _event("b", "tb", "provider/model-a", 0.8, "test LiODFB", False),
                _event("c", "tc", "provider/model-b", 0.7, "test LiPF6", False),
            ]
        )
        result = analyze_live_trajectories(trajectories)
        self.assertTrue(result.within_model_replication_available)
        self.assertEqual(set(result.within_model_numeric_dispersion), {"provider/model-a"})
        self.assertEqual(
            result.within_model_action_similarity["provider/model-a"]["pair_count"],
            1,
        )
        self.assertAlmostEqual(
            result.within_model_numeric_dispersion["provider/model-a"]["p_success"]["population_variance"],
            0.01,
        )

    def test_loader_reads_exported_trajectory_jsonl(self):
        trajectory = assemble_matbot_trajectories(
            [_event("a", "ta", "provider/model-a", 0.6, "test LiFSI")]
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            path.write_text(json.dumps(trajectory.to_json()) + "\n")
            loaded = load_matbot_trajectories(path)
        self.assertEqual(loaded, [trajectory])


if __name__ == "__main__":
    unittest.main()
