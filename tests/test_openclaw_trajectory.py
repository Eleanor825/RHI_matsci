import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.openclaw_trajectory import load_openclaw_matbot_trajectory


class OpenClawTrajectoryTests(unittest.TestCase):
    def test_imports_live_numeric_action_and_tool_trace(self):
        rows = [
            {"type": "session", "id": "session-1"},
            {"type": "model_change", "provider": "provider", "modelId": "model"},
            {
                "type": "message",
                "timestamp": "2026-08-22T12:00:00Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "visible"}]},
            },
            {
                "type": "message",
                "timestamp": "2026-08-22T12:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call-1", "name": "read", "arguments": {"path": "a.md"}}
                    ],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-08-22T12:00:02Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "isError": False,
                    "content": [{"type": "text", "text": "evidence"}],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-08-22T12:00:03Z",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "action": "run experiment",
                            "p_success": 0.7,
                            "expected_scientific_utility": 0.8,
                            "internal_uncertainty": 0.2,
                            "external_uncertainty": 0.3,
                            "evidence_value": 0.4,
                            "irreversible_risk": 0.1,
                            "estimated_cost": 0.2,
                            "rationale": "test",
                        }),
                    }],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            trajectory = load_openclaw_matbot_trajectory(path, system="ncm", stage="generation")
        self.assertTrue(trajectory.is_live_runtime)
        self.assertFalse(trajectory.complete)
        self.assertEqual(trajectory.steps[0].agent_backend, "provider/model")
        self.assertEqual(len(trajectory.steps[0].tool_trace), 1)
        self.assertEqual(trajectory.steps[0].action_text, "run experiment")
        self.assertEqual(
            trajectory.steps[0].uncertainty_output["reported_internal_uncertainty"],
            0.2,
        )
        self.assertNotIn("internal_variance", trajectory.steps[0].uncertainty_output)
        self.assertNotIn("selected_route", trajectory.steps[0].uncertainty_output)


if __name__ == "__main__":
    unittest.main()
