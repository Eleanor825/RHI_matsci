import unittest

from harness_matsci.portfolio_knowledge_gradient import (
    PortfolioKnowledgeGradientRow,
    fit_portfolio_knowledge_gradient_model,
    portfolio_kg_features,
    realized_single_acquisition_portfolio_gain,
)


class PortfolioKnowledgeGradientTest(unittest.TestCase):
    def test_realized_gain_accounts_for_rank_change_and_evidence_cost(self):
        value = realized_single_acquisition_portfolio_gain(
            current_scores=[0.9, 0.7, 0.4],
            current_action_gains=[0.8, -0.2, 0.5],
            candidate_index=2,
            future_score=0.85,
            future_action_gain=0.6,
            budget=2,
            charged_evidence_cost=0.05,
        )
        self.assertAlmostEqual(value, 0.75)

    def test_features_capture_decision_boundary_crossing(self):
        features = portfolio_kg_features(
            current_score=0.2,
            shadow_price=0.5,
            structural_mean=0.1,
            structural_variance=0.04,
            internal_variance=0.01,
            external_variance=0.02,
            interaction_variance=0.01,
            positive_probability=0.75,
            evidence_cost=0.1,
            future_score_cells=[[0.4, 0.6], [0.7, 0.2]],
        )
        self.assertAlmostEqual(features["future_crossing_probability"], 0.5)
        self.assertAlmostEqual(features["distance_to_shadow"], -0.3)

    def test_model_outputs_numeric_incremental_gain(self):
        rows = []
        for index in range(60):
            signal = index / 60.0
            rows.append(
                PortfolioKnowledgeGradientRow(
                    record_id=str(index),
                    benchmark="task",
                    transition="base->screening",
                    features={"signal": signal, "cost": 0.1},
                    realized_incremental_gain=signal - 0.5,
                )
            )
        model = fit_portfolio_knowledge_gradient_model(
            rows,
            seed=7,
            replicas=2,
            trees=12,
            min_samples_leaf=2,
        )
        prediction = model.predict(
            record_id="held",
            benchmark="task",
            transition="base->screening",
            features={"signal": 0.9, "cost": 0.1},
        )
        self.assertGreater(prediction.expected_incremental_gain, 0.0)
        self.assertGreaterEqual(prediction.internal_variance, 0.0)
        self.assertGreaterEqual(
            prediction.predictive_variance,
            prediction.internal_variance,
        )


if __name__ == "__main__":
    unittest.main()
