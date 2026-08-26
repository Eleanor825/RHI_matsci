import unittest

from harness_matsci.hurdle_gain_calibration import HurdleGainPrediction
from harness_matsci.two_stage_policy import (
    TwoStagePolicy,
    evaluate_two_stage_policy,
    select_execution_threshold,
    select_two_stage_policy,
)


def prediction(index, score):
    return HurdleGainPrediction(
        record_id=str(index),
        benchmark="task",
        mean_gain=score,
        p_positive_gain=0.9 if score > 0 else 0.1,
        expected_shortfall=0.0,
        tail_risk_score=score,
        irreducible_variance=0.1,
        calibrated_internal_variance=0.01,
        calibrated_external_variance=0.02,
        calibrated_variance=0.13,
        positive_component_mean=0.4,
        negative_component_mean=-0.4,
    )


class TwoStagePolicyTests(unittest.TestCase):
    def test_execution_threshold_maximizes_safe_population_gain(self):
        predictions = [prediction(index, 1.0 - index / 10.0) for index in range(10)]
        labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        gains = [0.5, 0.4, 0.3, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        threshold = select_execution_threshold(
            predictions,
            labels,
            gains,
            coverages=(0.1, 0.2, 0.3, 0.5, 1.0),
            max_empirical_risk=0.20,
        )
        self.assertEqual(threshold.target_coverage, 0.3)
        self.assertAlmostEqual(threshold.population_gain, 0.12)
        self.assertEqual(threshold.selective_risk, 0.0)

    def test_value_routing_acquires_records_that_tool_rescues(self):
        count = 100
        base = [prediction(index, -0.1) for index in range(count)]
        tool = [prediction(index, 0.3 if index < 30 else -0.2) for index in range(count)]
        acquisition = [1.0 - index / count for index in range(count)]
        labels = [1] * 30 + [0] * 70
        gains = [0.5] * 30 + [-0.5] * 70
        policy = select_two_stage_policy(
            acquisition,
            base,
            tool,
            labels,
            gains,
            tool_cost=0.02,
            coverages=(0.1, 0.2, 0.3, 0.5),
            execution_coverages=(0.1, 0.2, 0.3, 0.5),
        )
        self.assertEqual(policy.target_acquisition_coverage, 0.3)
        result = evaluate_two_stage_policy(
            policy,
            acquisition,
            base,
            tool,
            labels,
            gains,
            tool_cost=0.02,
            alpha=0.20,
            delta=0.05,
        )
        self.assertTrue(result.certified)
        self.assertEqual(result.executed_count, 30)
        self.assertGreater(result.population_gain_after_cost, 0.0)

    def test_verify_before_execute_disables_base_execution(self):
        count = 100
        base = [prediction(index, 0.9) for index in range(count)]
        tool = [prediction(index, 0.8 if index < 20 else -0.2) for index in range(count)]
        acquisition = [1.0 - index / count for index in range(count)]
        labels = [1] * 20 + [0] * 80
        gains = [0.5] * 20 + [-0.5] * 80
        policy = select_two_stage_policy(
            acquisition,
            base,
            tool,
            labels,
            gains,
            tool_cost=0.02,
            coverages=(0.1, 0.2, 0.5),
            base_execution_coverages=(0.0,),
            tool_execution_coverages=(0.1, 0.2, 0.5),
        )
        self.assertEqual(policy.target_base_execution_coverage, 0.0)
        result = evaluate_two_stage_policy(
            policy,
            acquisition,
            base,
            tool,
            labels,
            gains,
            tool_cost=0.02,
            alpha=0.20,
            delta=0.05,
        )
        self.assertEqual(result.executed_count, 20)
        self.assertEqual(result.errors, 0)

    def test_rank_budget_preserves_coverage_under_score_shift(self):
        count = 100
        base = [prediction(index, -0.2) for index in range(count)]
        tool = [prediction(index, 1.0 - index / count) for index in range(count)]
        policy = TwoStagePolicy(
            acquisition_threshold=10.0,
            target_acquisition_coverage=0.20,
            base_execution_threshold=float("inf"),
            target_base_execution_coverage=0.0,
            tool_execution_threshold=10.0,
            target_tool_execution_coverage=0.10,
            feedback_population_gain=0.0,
            feedback_selective_risk=0.0,
            selection_mode="rank_budget",
        )
        shifted_scores = [-100.0 - index for index in range(count)]
        result = evaluate_two_stage_policy(
            policy,
            shifted_scores,
            base,
            tool,
            [1] * count,
            [0.1] * count,
            tool_cost=0.02,
            alpha=0.20,
            delta=0.05,
        )
        self.assertEqual(result.acquired_count, 20)
        self.assertEqual(result.executed_count, 10)


if __name__ == "__main__":
    unittest.main()
