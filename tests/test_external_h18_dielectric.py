import unittest

from harness_matsci.external_h18_dielectric_tools import _pair_utility


class ExternalH18DielectricTests(unittest.TestCase):
    def test_pair_utility_is_directional_and_normalized(self):
        values = (1.0, 2.0, 3.0, 4.0)
        self.assertGreater(_pair_utility(values, 4.0, 1.0), 0.0)
        self.assertEqual(_pair_utility(values, 1.0, 4.0), 0.0)
        self.assertLessEqual(_pair_utility(values, 4.0, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
