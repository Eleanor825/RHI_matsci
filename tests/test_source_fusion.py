import unittest

from harness_matsci.source_fusion import (
    SourceFusionObservation,
    fit_source_reliability_fuser,
)


class SourceFusionTests(unittest.TestCase):
    def test_fuser_learns_reliable_source(self):
        observations = []
        for index in range(200):
            gain = -0.5 + index / 199.0
            observations.append(
                SourceFusionObservation(
                    record_id=str(index),
                    benchmark="task",
                    source_names=("weak_a", "weak_b", "strong"),
                    predicted_gain=(
                        (-gain, 0.0, gain - 0.01),
                        (-gain + 0.02, 0.1, gain + 0.01),
                        (-gain - 0.02, -0.1, gain),
                    ),
                    gain=gain,
                )
            )
        model = fit_source_reliability_fuser(observations)
        weights = model.parameters_by_task["task"].source_weights
        self.assertGreater(weights["strong"], weights["weak_a"])
        self.assertGreater(weights["strong"], weights["weak_b"])
        self.assertGreater(weights["strong"], 0.65)
        prediction = model.predict_one(observations[-1])
        self.assertGreater(prediction.mean_gain, 0.4)
        self.assertGreaterEqual(prediction.internal_variance, 0.0)
        self.assertGreaterEqual(prediction.external_variance, 0.0)
        self.assertGreaterEqual(prediction.interaction_variance, 0.0)
        self.assertGreaterEqual(prediction.internal_boundary_instability, 0.0)
        self.assertGreaterEqual(prediction.external_boundary_instability, 0.0)
        self.assertAlmostEqual(
            prediction.total_variance,
            prediction.internal_variance
            + prediction.external_variance
            + prediction.interaction_variance,
        )

    def test_precalibrated_sources_fit_weights_without_second_affine_map(self):
        observations = []
        for index in range(120):
            gain = -0.5 + index / 119.0
            observations.append(
                SourceFusionObservation(
                    record_id=str(index),
                    benchmark="task",
                    source_names=("strong", "weak"),
                    predicted_gain=(
                        (gain - 0.01, -gain - 0.01),
                        (gain, -gain),
                        (gain + 0.01, -gain + 0.01),
                    ),
                    gain=gain,
                )
            )
        model = fit_source_reliability_fuser(observations, fit_affine=False)
        parameters = model.parameters_by_task["task"]
        self.assertGreater(parameters.source_weights["strong"], 0.9)
        self.assertEqual(parameters.source_slopes["strong"], 1.0)
        self.assertEqual(parameters.source_biases["strong"], 0.0)


if __name__ == "__main__":
    unittest.main()
