from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.trajectory_dataset import build_dataset


class TrajectoryDatasetTests(unittest.TestCase):
    def test_packages_action_and_grouped_views_without_oracle_input_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "actions.jsonl"
            source.write_text(json.dumps({
                "record_id": "a1",
                "task": "materials_pairwise_preference",
                "trace_id": "trace-1",
                "group_id": "group-1",
                "event_seq": 0,
                "visible_context": "Candidate A; log10(K_VRH)=1.2",
                "candidate_action": "Choose candidate A",
                "action_type": "choose_candidate",
                "evidence": ["log10(K_VRH)=1.2"],
                "context_features": {"performance_score": 0.9, "space_group": 10},
                "uncertainty_signals": {"performance_score": 0.9, "source_risk": 0.1},
                "outcome_success": True,
                "metric_value": 0.8,
                "label_source": "test",
            }) + "\n")
            out = root / "out"
            manifest = build_dataset([source], out)
            self.assertEqual(manifest["records"], 1)
            self.assertEqual(manifest["trajectories"], 1)
            with gzip.open(out / "actions.jsonl.gz", "rt") as handle:
                action = json.loads(next(handle))
            self.assertNotIn("performance_score", action["model_input"]["context_features"])
            self.assertNotIn("performance_score", action["model_input"]["uncertainty_signals"])
            self.assertIn("label", action["outcome"])
            with gzip.open(out / "trajectories.jsonl.gz", "rt") as handle:
                trajectory = json.loads(next(handle))
            self.assertEqual(trajectory["action_count"], 1)
            self.assertEqual(trajectory["actions"][0]["decision_unit_id"], "a1")


if __name__ == "__main__":
    unittest.main()
