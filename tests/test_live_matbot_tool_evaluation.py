import unittest

from experiments.sd_worth.evaluate_live_matbot_tool_trajectories import (
    _clopper_pearson_upper,
    _ece,
)


class LiveMatBotToolEvaluationTests(unittest.TestCase):
    def test_zero_error_bound_shrinks_with_sample_size(self):
        self.assertLess(_clopper_pearson_upper(0, 100), _clopper_pearson_upper(0, 10))

    def test_perfect_calibration_has_zero_ece(self):
        self.assertAlmostEqual(_ece([0.0, 1.0], [0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
