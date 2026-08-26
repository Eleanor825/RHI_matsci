import unittest

from harness_matsci.frozen_policy_certificate import (
    certify_frozen_policy,
    select_feedback_policy,
)


class FrozenPolicyCertificateTests(unittest.TestCase):
    def test_feedback_selection_and_independent_certificate(self):
        labels = [1] * 30 + [0] * 70
        gains = [0.4] * 30 + [-0.5] * 70
        scores = [1.0 - index / 100.0 for index in range(100)]
        policy = select_feedback_policy(
            {"tail": scores},
            labels,
            gains,
            {"tail": 0.02},
            coverages=(0.1, 0.2, 0.3, 0.5),
        )
        self.assertEqual(policy.method, "tail")
        self.assertEqual(policy.selected_count, 30)
        certificate = certify_frozen_policy(
            policy,
            scores,
            labels,
            gains,
            tool_cost=0.02,
            alpha=0.20,
            delta=0.05,
        )
        self.assertTrue(certificate.certified)
        self.assertGreater(certificate.population_gain_after_cost, 0.0)

    def test_no_positive_feedback_policy_returns_no_op(self):
        policy = select_feedback_policy(
            {"bad": [0.9, 0.8, 0.7]},
            [0, 0, 0],
            [-0.2, -0.2, -0.2],
            {"bad": 0.0},
            coverages=(0.5, 1.0),
        )
        self.assertEqual(policy.method, "no_op")
        self.assertEqual(policy.selected_count, 0)


if __name__ == "__main__":
    unittest.main()
