import unittest

from harness_matsci.decision_instability import decision_boundary_instability


class DecisionInstabilityTests(unittest.TestCase):
    def test_sign_consistent_ensemble_has_zero_boundary_instability(self):
        result = decision_boundary_instability([0.1, 0.4, 0.9])
        self.assertEqual(result.normalized_vote_entropy, 0.0)
        self.assertEqual(result.boundary_instability, 0.0)

    def test_sign_flip_is_larger_when_value_range_is_larger(self):
        narrow = decision_boundary_instability([-0.1, 0.1])
        wide = decision_boundary_instability([-1.0, 1.0])
        self.assertEqual(narrow.normalized_vote_entropy, 1.0)
        self.assertGreater(wide.boundary_instability, narrow.boundary_instability)

    def test_weighted_sources_respect_reliability_mass(self):
        result = decision_boundary_instability([-1.0, 1.0], weights=[0.9, 0.1])
        self.assertAlmostEqual(result.positive_fraction, 0.1)
        self.assertGreater(result.boundary_instability, 0.0)


if __name__ == "__main__":
    unittest.main()
