import unittest

from harness_matsci.selective_execution_policy import (
    TaskBalancedExecutionPolicy,
    fit_task_balanced_execution_policy,
)


class SelectiveExecutionPolicyTest(unittest.TestCase):
    def test_task_balanced_selection_applies_threshold_and_budget_per_task(self):
        policy = TaskBalancedExecutionPolicy(0.8, 0.5)
        selected = policy.select(
            decision_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            worthiness_probabilities=[0.9, 0.7, 0.95, 0.99, 0.85, 0.4],
            benchmarks=["a", "a", "a", "b", "b", "b"],
            record_ids=["0", "1", "2", "3", "4", "5"],
        )
        self.assertEqual(selected, (0, 2, 3, 4))

    def test_fit_returns_none_when_no_threshold_is_risk_certified(self):
        policy, evidence = fit_task_balanced_execution_policy(
            candidate_thresholds=(0.5, 0.9),
            decision_scores=[1.0] * 40,
            worthiness_probabilities=[0.95] * 40,
            realized_gains=[-1.0] * 40,
            benchmarks=["a"] * 20 + ["b"] * 20,
            record_ids=[str(index) for index in range(40)],
            budget_fraction=0.5,
            alpha=0.1,
            delta=0.05,
        )
        self.assertIsNone(policy)
        self.assertTrue(all(not row.certified for row in evidence))


if __name__ == "__main__":
    unittest.main()
