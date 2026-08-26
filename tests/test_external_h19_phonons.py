import unittest

from harness_matsci.external_h19_phonon_pairwise import DATASET, TASK
from harness_matsci.external_h19_phonon_tools import ExternalH19PhononEvidenceEnvironment


class ExternalH19PhononTests(unittest.TestCase):
    def test_environment_targets_phonon_dataset(self):
        self.assertEqual(ExternalH19PhononEvidenceEnvironment.dataset_name, DATASET)
        self.assertEqual(ExternalH19PhononEvidenceEnvironment.task_name, TASK)
        self.assertIn("phonon", ExternalH19PhononEvidenceEnvironment.source_tool_name)


if __name__ == "__main__":
    unittest.main()
