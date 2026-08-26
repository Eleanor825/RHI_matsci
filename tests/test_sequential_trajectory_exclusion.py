import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.sequential_trajectory_export import (
    _load_excluded_fingerprints,
    export_sequential_benchmark_trajectories,
    scientific_record_fingerprint,
)


class SequentialTrajectoryExclusionTest(unittest.TestCase):
    def test_scientific_fingerprint_is_stable_to_whitespace(self):
        first = scientific_record_fingerprint("task", "a  b", "do  x")
        second = scientific_record_fingerprint("task", "a b", "do x")
        self.assertEqual(first, second)

    def test_exclusion_reads_adaptive_trajectory_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            payload = {
                "benchmark": "task",
                "post_acquisition_observed_stages": [
                    {
                        "stage": "base",
                        "visible_observation": {
                            "record": {
                                "visible_context": "context",
                                "candidate_action": "action",
                            }
                        },
                    }
                ],
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            self.assertEqual(
                _load_excluded_fingerprints((path,)),
                {scientific_record_fingerprint("task", "context", "action")},
            )

    def test_unknown_task_count_override_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            export_sequential_benchmark_trajectories(
                "/tmp/does-not-matter",
                "/tmp/does-not-matter.jsonl",
                per_task_counts={"not_a_task": 1},
            )


if __name__ == "__main__":
    unittest.main()
