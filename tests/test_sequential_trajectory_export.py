import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.sequential_trajectory_export import export_sequential_benchmark_trajectories


RAW = Path("/Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks")


@unittest.skipUnless(RAW.exists(), "historical material-discovery data are not available")
class SequentialTrajectoryExportTests(unittest.TestCase):
    def test_exports_equal_task_counts_and_hidden_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectories.jsonl"
            manifest = export_sequential_benchmark_trajectories(RAW, output, per_task=2)
            self.assertEqual(manifest["total_trajectories"], 6)
            self.assertEqual(set(manifest["task_counts"].values()), {2})
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(all(len(row["stages"]) == 3 for row in rows))
            self.assertTrue(all("hidden_outcome_for_evaluation_only" in row for row in rows))
            for row in rows:
                visible = json.dumps(row["stages"], sort_keys=True)
                self.assertNotIn('"worthy"', visible)
                self.assertNotIn('"success_label"', visible)


if __name__ == "__main__":
    unittest.main()
