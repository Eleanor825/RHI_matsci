import unittest

from harness_matsci.evidence_value import (
    EvidenceValueObservation,
    cross_fitted_evidence_value_predictions,
    evidence_value_observation,
    fit_evidence_value_model,
    realized_counterfactual_evidence_value,
)
from harness_matsci.hurdle_gain_calibration import HurdleGainPrediction
from harness_matsci.source_fusion import SourceFusionObservation


def _prediction(record_id: str, score: float) -> HurdleGainPrediction:
    return HurdleGainPrediction(
        record_id=record_id,
        benchmark="task",
        mean_gain=score,
        p_positive_gain=0.7,
        expected_shortfall=0.2,
        tail_risk_score=score,
        irreducible_variance=0.1,
        calibrated_internal_variance=0.02,
        calibrated_external_variance=0.03,
        calibrated_variance=0.15,
        positive_component_mean=0.1,
        negative_component_mean=-0.4,
    )


class EvidenceValueTests(unittest.TestCase):
    def test_realized_value_rewards_a_tool_that_prevents_a_bad_action(self):
        base = _prediction("record", 0.1)
        tool = _prediction("record", -0.1)
        value = realized_counterfactual_evidence_value(
            base,
            tool,
            realized_gain=-0.8,
            tool_cost=0.05,
        )
        self.assertAlmostEqual(value, 0.75)

    def test_realized_value_charges_unhelpful_tool_cost(self):
        base = _prediction("record", -0.1)
        tool = _prediction("record", -0.2)
        value = realized_counterfactual_evidence_value(
            base,
            tool,
            realized_gain=0.5,
            tool_cost=0.05,
        )
        self.assertAlmostEqual(value, -0.05)

    def test_realized_value_respects_policy_execution_thresholds(self):
        base = _prediction("record", 0.3)
        tool = _prediction("record", 0.6)
        value = realized_counterfactual_evidence_value(
            base,
            tool,
            realized_gain=0.5,
            tool_cost=0.05,
            base_execution_threshold=0.4,
            tool_execution_threshold=0.5,
        )
        self.assertAlmostEqual(value, 0.45)

    def test_model_learns_external_variance_value_pattern(self):
        observations = []
        for index in range(200):
            external_variance = (index % 20) / 20.0
            observations.append(
                EvidenceValueObservation(
                    record_id=str(index),
                    benchmark="task",
                    base_mean_gain=0.05,
                    base_p_positive_gain=0.6,
                    base_expected_shortfall=0.2,
                    base_tail_risk_score=-0.05,
                    base_internal_variance=0.01,
                    base_external_variance=external_variance,
                    tool_cost=0.02,
                    realized_value=external_variance - 0.4,
                )
            )
        model = fit_evidence_value_model(observations, seed=7, replicas=3, trees=32)
        low = model.predict_one(observations[0])
        high = model.predict_one(observations[-1])
        self.assertGreater(high.expected_value, low.expected_value)
        self.assertGreater(high.positive_value_probability, low.positive_value_probability)
        self.assertGreaterEqual(high.predictive_variance, high.residual_variance)
        self.assertEqual(high.to_json()["record_id"], high.record_id)

    def test_regularized_linear_model_learns_external_variance_pattern(self):
        observations = []
        for index in range(200):
            external_variance = (index % 20) / 20.0
            observations.append(
                EvidenceValueObservation(
                    record_id=str(index),
                    benchmark="task",
                    base_mean_gain=0.05,
                    base_p_positive_gain=0.6,
                    base_expected_shortfall=0.2,
                    base_tail_risk_score=-0.05,
                    base_internal_variance=0.01,
                    base_external_variance=external_variance,
                    tool_cost=0.02,
                    realized_value=external_variance - 0.4,
                )
            )
        model = fit_evidence_value_model(
            observations,
            seed=11,
            replicas=3,
            model_family="regularized_linear",
        )
        low = model.predict_one(observations[0])
        high = model.predict_one(observations[-1])
        self.assertGreater(high.expected_value, low.expected_value)
        self.assertGreater(high.positive_value_probability, low.positive_value_probability)

    def test_observation_builder_uses_numeric_harness_output(self):
        base = _prediction("record", 0.1)
        tool = _prediction("record", -0.1)
        source = SourceFusionObservation(
            record_id="record",
            benchmark="task",
            source_names=("agent_prior", "external"),
            predicted_gain=((-0.2, 0.3), (-0.1, 0.4), (0.0, 0.5)),
            gain=-0.8,
        )
        observation = evidence_value_observation(
            base,
            tool,
            realized_gain=-0.8,
            tool_cost=0.05,
            source_observation=source,
        )
        self.assertAlmostEqual(observation.realized_value, 0.75)
        self.assertEqual(observation.base_external_variance, 0.03)
        self.assertEqual(observation.base_source_sign_disagreement, 1.0)
        self.assertGreater(observation.base_source_range, 0.0)
        self.assertAlmostEqual(observation.base_internal_prior_mean, -0.1)
        self.assertGreater(observation.base_internal_prior_variance, 0.0)
        self.assertAlmostEqual(observation.base_external_mean, 0.4)
        self.assertAlmostEqual(observation.base_internal_external_gap, -0.5)

    def test_cross_fitted_predictions_cover_every_record(self):
        observations = []
        for task in ("task_a", "task_b"):
            for index in range(60):
                observations.append(
                    EvidenceValueObservation(
                        record_id=f"{task}-{index}",
                        benchmark=task,
                        base_mean_gain=index / 100.0,
                        base_p_positive_gain=0.4,
                        base_expected_shortfall=0.1,
                        base_tail_risk_score=-0.05,
                        base_internal_variance=0.01,
                        base_external_variance=index / 1000.0,
                        tool_cost=0.02,
                        realized_value=0.2 if index % 3 == 0 else -0.05,
                    )
                )
        predictions = cross_fitted_evidence_value_predictions(
            observations,
            seed=17,
            folds=3,
            replicas=2,
            trees=12,
        )
        self.assertEqual(len(predictions), len(observations))
        self.assertEqual(
            [prediction.record_id for prediction in predictions],
            [observation.record_id for observation in observations],
        )

    def test_batch_prediction_matches_single_prediction(self):
        observations = []
        for index in range(40):
            observations.append(
                EvidenceValueObservation(
                    record_id=str(index),
                    benchmark="task",
                    base_mean_gain=index / 100.0,
                    base_p_positive_gain=0.5,
                    base_expected_shortfall=0.1,
                    base_tail_risk_score=0.0,
                    base_internal_variance=0.01,
                    base_external_variance=index / 1000.0,
                    tool_cost=0.02,
                    realized_value=0.1 if index % 2 else -0.02,
                )
            )
        model = fit_evidence_value_model(
            observations,
            seed=19,
            replicas=2,
            trees=12,
        )
        batch = model.predict(observations[:5])
        single = [model.predict_one(row) for row in observations[:5]]
        for batch_row, single_row in zip(batch, single):
            self.assertEqual(batch_row.record_id, single_row.record_id)
            self.assertEqual(batch_row.benchmark, single_row.benchmark)
            for field in (
                "expected_value",
                "internal_variance",
                "residual_variance",
                "predictive_variance",
                "positive_value_probability",
            ):
                self.assertAlmostEqual(
                    getattr(batch_row, field), getattr(single_row, field), places=12
                )

    def test_tool_and_context_features_are_numeric_model_inputs(self):
        names = __import__(
            "harness_matsci.evidence_value", fromlist=["_FEATURE_NAMES"]
        )._FEATURE_NAMES
        self.assertIn("candidate_tool_high_fidelity", names)
        self.assertIn("context_ood_score", names)
        self.assertIn("base_source_interaction_variance", names)


if __name__ == "__main__":
    unittest.main()
