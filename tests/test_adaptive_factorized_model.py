import unittest

from harness_matsci.adaptive_factorized_model import (
    FactorizedGainTrainingRow,
    fit_adaptive_factorized_gain_model,
)


class AdaptiveFactorizedModelTest(unittest.TestCase):
    def test_factorized_prediction_is_numeric_and_cost_aware(self):
        rows = []
        for index in range(80):
            success = int(index % 4 == 0 or index > 60)
            rows.append(
                FactorizedGainTrainingRow(
                    record_id=str(index),
                    benchmark="task",
                    features={"signal": index / 80.0, "noise": (index % 7) / 7.0},
                    success=success,
                    normalized_utility=0.7 + 0.2 * (index / 80.0),
                    normalized_action_cost=0.2,
                    failure_harm=0.4,
                )
            )
        model = fit_adaptive_factorized_gain_model(
            rows,
            seed=11,
            replicas=2,
            trees=12,
            min_samples_leaf=2,
        )
        views = [{"signal": 0.8, "noise": value} for value in (0.1, 0.3, 0.5)]
        cheap = model.predict_crossed(
            record_id="held",
            benchmark="task",
            feature_views=views,
            normalized_action_cost=0.2,
            failure_harm=0.4,
            cumulative_evidence_cost=0.0,
        )
        expensive = model.predict_crossed(
            record_id="held",
            benchmark="task",
            feature_views=views,
            normalized_action_cost=0.2,
            failure_harm=0.4,
            cumulative_evidence_cost=0.5,
        )
        self.assertGreater(cheap.expected_net_gain, expensive.expected_net_gain)
        self.assertAlmostEqual(
            cheap.gain_internal_variance
            + cheap.gain_external_variance
            + cheap.gain_interaction_variance
            + cheap.aleatoric_gain_variance,
            cheap.predictive_gain_variance,
        )
        self.assertNotIn("route", cheap.to_json())


if __name__ == "__main__":
    unittest.main()
