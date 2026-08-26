from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.prompt_trajectories import generate_prompt_trajectories


class PromptTrajectoryTests(unittest.TestCase):
    def test_generation_is_balanced_and_prompt_does_not_include_outcomes(self) -> None:
        data_dir = Path('/Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks')
        if not data_dir.exists():
            self.skipTest('local historical materials data unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_prompt_trajectories(data_dir, tmp, n_per_task=2, seed=3, max_steps=3)
            self.assertEqual(manifest['trajectory_count'], 6)
            rows = [json.loads(line) for line in (Path(tmp) / 'trajectories.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            for row in rows:
                prompt = '\n'.join(step['user_prompt'] for step in row['steps'])
                self.assertNotIn('outcome_success', prompt)
                self.assertNotIn('metric_value', prompt)
                self.assertNotIn('utility=', prompt)
                self.assertTrue(row['prompt_generated'])
                self.assertFalse(row['llm_generated'])
                self.assertIn('outcome', row)


if __name__ == '__main__':
    unittest.main()
