import unittest

from harness_matsci.external_h20_perovskite_pairwise import DATASET, TASK
from harness_matsci.external_h20_perovskite_tools import (
    ExternalH20PerovskiteEvidenceEnvironment,
)


class ExternalH20PerovskiteTests(unittest.TestCase):
    def test_environment_uses_frozen_task_and_dataset(self):
        self.assertEqual(ExternalH20PerovskiteEvidenceEnvironment.dataset_name, DATASET)
        self.assertEqual(ExternalH20PerovskiteEvidenceEnvironment.task_name, TASK)
        self.assertIn("perovskite", ExternalH20PerovskiteEvidenceEnvironment.source_tool_name)


if __name__ == "__main__":
    unittest.main()
