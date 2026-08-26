import unittest

from harness_matsci.adaptive_gain_harness import (
    conservative_score,
    counterfactual_evidence_value_cells,
    crossed_numeric_estimate,
    leave_one_out_topk_shadow_prices,
    sunk_cost_neutral_score_cells,
)


class AdaptiveGainHarnessTest(unittest.TestCase):
    def test_crossed_components_sum_to_predictive_variance(self):
        estimate = crossed_numeric_estimate(
            [[0.1, 0.2, 0.4], [0.3, 0.5, 0.9]],
            aleatoric_variance=0.07,
        )
        self.assertAlmostEqual(
            estimate.internal_variance
            + estimate.external_variance
            + estimate.interaction_variance,
            estimate.epistemic_variance,
        )
        self.assertAlmostEqual(
            estimate.epistemic_variance + estimate.aleatoric_variance,
            estimate.predictive_variance,
        )
        self.assertEqual(estimate.model_replicas, 2)
        self.assertEqual(estimate.evidence_replicas, 3)

    def test_counterfactual_value_charges_cost_and_uses_best_current_action(self):
        cells = counterfactual_evidence_value_cells(
            [0.2, -0.1],
            [[0.5, -0.2], [0.3, -0.4]],
            evidence_cost=0.1,
            cost_weight=0.5,
        )
        self.assertEqual(cells, [[0.25, -0.25], [0.25, -0.05]])

    def test_conservative_score_decreases_with_variance(self):
        self.assertGreater(
            conservative_score(0.4, 0.01, 1.0),
            conservative_score(0.4, 0.25, 1.0),
        )

    def test_counterfactual_value_respects_paid_cost_abstention(self):
        cells = counterfactual_evidence_value_cells(
            [-0.2, -0.3],
            [[-0.4, 0.2], [-0.5, -0.6]],
            evidence_cost=0.0,
            cost_weight=0.15,
            current_abstention_value=-0.01,
            future_abstention_value=-0.02,
        )
        self.assertAlmostEqual(cells[0][0], -0.01)
        self.assertAlmostEqual(cells[0][1], 0.21)
        self.assertAlmostEqual(cells[1][0], -0.01)
        self.assertAlmostEqual(cells[1][1], -0.01)

    def test_numeric_core_contains_no_route_decision(self):
        estimate = crossed_numeric_estimate([[0.1, 0.2], [0.2, 0.3]])
        self.assertNotIn("route", estimate.to_json())
        self.assertNotIn("selected_route", estimate.to_json())

    def test_sunk_cost_neutral_scores_remove_paid_evidence_cost(self):
        scores = sunk_cost_neutral_score_cells(
            [[0.10, 0.20], [0.30, 0.40]],
            cumulative_evidence_cost=0.5,
            cost_weight=0.2,
            lower_bound_radius=0.05,
        )
        expected = [[0.15, 0.25], [0.35, 0.45]]
        for actual_row, expected_row in zip(scores, expected):
            for actual, target in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, target)

    def test_leave_one_out_shadow_is_exact_topk_replacement_value(self):
        shadows = leave_one_out_topk_shadow_prices(
            [0.9, 0.7, 0.4, -0.2],
            budget=2,
        )
        self.assertEqual(shadows, [0.4, 0.4, 0.7, 0.7])


if __name__ == "__main__":
    unittest.main()
