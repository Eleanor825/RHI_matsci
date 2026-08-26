import unittest

from harness_matsci.hurdle_gain_calibration import HurdleGainPrediction
from harness_matsci.multi_tool_policy import (
    evaluate_multi_tool_policy,
    select_certified_tool_contract,
    select_multi_tool_policy,
)


def prediction(index, score):
    return HurdleGainPrediction(
        record_id=str(index),
        benchmark="task",
        mean_gain=score,
        p_positive_gain=0.9,
        expected_shortfall=0.0,
        tail_risk_score=score,
        irreducible_variance=0.1,
        calibrated_internal_variance=0.01,
        calibrated_external_variance=0.02,
        calibrated_variance=0.13,
        positive_component_mean=0.4,
        negative_component_mean=-0.4,
    )


class MultiToolPolicyTests(unittest.TestCase):
    def test_numeric_values_select_different_tools_without_route_labels(self):
        count = 100
        scores = {
            "retrieve": [0.5 if index < 15 else -0.2 for index in range(count)],
            "simulate": [0.6 if 15 <= index < 30 else -0.3 for index in range(count)],
        }
        predictions = {
            "retrieve": [prediction(index, 0.8 if index < 15 else -0.2) for index in range(count)],
            "simulate": [prediction(index, 0.8 if 15 <= index < 30 else -0.2) for index in range(count)],
        }
        labels = [1] * 30 + [0] * 70
        gains = [0.5] * 30 + [-0.5] * 70
        policy = select_multi_tool_policy(
            scores,
            predictions,
            labels,
            gains,
            tool_costs={"retrieve": 0.01, "simulate": 0.02},
            acquisition_coverages=(0.1, 0.2, 0.3, 0.5),
            execution_coverages=(0.1, 0.2, 0.3, 0.5),
            selection_mode="rank_budget",
        )
        result = evaluate_multi_tool_policy(
            policy,
            scores,
            predictions,
            labels,
            gains,
            tool_costs={"retrieve": 0.01, "simulate": 0.02},
            alpha=0.20,
            delta=0.05,
        )
        self.assertTrue(result.certified)
        self.assertEqual(result.acquired_count, 30)
        self.assertEqual(result.executed_count, 30)
        self.assertEqual(result.selected_tool_counts["retrieve"], 15)
        self.assertEqual(result.selected_tool_counts["simulate"], 15)
        self.assertGreater(result.population_gain_after_cost, 0.0)

    def test_acceptance_set_prunes_unsafe_tool_with_familywise_certificate(self):
        count = 200
        feedback_scores = {
            "cheap": [0.8 if 40 <= index < 80 else -0.2 for index in range(count)],
            "safe": [0.6 if index < 40 else -0.3 for index in range(count)],
        }
        feedback_predictions = {
            "cheap": [prediction(index, 0.8 if 40 <= index < 80 else -0.2) for index in range(count)],
            "safe": [prediction(index, 0.8 if index < 40 else -0.2) for index in range(count)],
        }
        feedback_labels = [1] * 80 + [0] * 120
        feedback_gains = [0.5] * 80 + [-0.5] * 120
        acceptance_scores = feedback_scores
        acceptance_predictions = feedback_predictions
        acceptance_labels = [0] * count
        acceptance_labels[:40] = [1] * 40
        acceptance_gains = [0.5 if label else -0.5 for label in acceptance_labels]
        contract = select_certified_tool_contract(
            feedback_scores,
            acceptance_scores,
            feedback_predictions,
            acceptance_predictions,
            feedback_labels,
            acceptance_labels,
            feedback_gains,
            acceptance_gains,
            tool_costs={"cheap": 0.001, "safe": 0.02},
            acquisition_coverages=(0.2,),
            execution_coverages=(0.2,),
            alpha=0.20,
            familywise_delta=0.05,
            baseline_tool="safe",
            baseline_weight=0.90,
            selection_mode="rank_budget",
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract.selected_tools, ("safe",))
        self.assertTrue(contract.acceptance.certified)


if __name__ == "__main__":
    unittest.main()
