import unittest

from harness_matsci.continuous_gain_calibration import (
    ContinuousGainObservation,
    fit_continuous_gain_calibrator,
)


class ContinuousGainCalibrationTests(unittest.TestCase):
    def test_recovers_positive_source_variance_weight(self):
        observations = []
        for index in range(120):
            external = (index % 6) / 20.0
            mean_gain = -0.4 + 0.8 * (index / 119.0)
            signed_noise = (1 if index % 2 else -1) * (0.02 + external)
            observations.append(
                ContinuousGainObservation(
                    record_id=str(index),
                    benchmark="task",
                    mean_gain=mean_gain,
                    internal_variance=0.0,
                    external_variance=external * external,
                    gain=mean_gain + signed_noise,
                )
            )
        model = fit_continuous_gain_calibrator(observations)
        self.assertGreater(model.parameters_by_task["task"].external_weight, 0.0)
        predictions = model.predict(observations)
        self.assertGreater(predictions[-1].action_worthiness, predictions[0].action_worthiness)
        prediction = predictions[-1]
        self.assertAlmostEqual(
            prediction.calibrated_variance,
            prediction.irreducible_variance
            + prediction.calibrated_internal_variance
            + prediction.calibrated_external_variance
            + prediction.calibrated_interaction_variance,
        )
        self.assertGreaterEqual(prediction.expected_shortfall, 0.0)
        self.assertEqual(prediction.to_json()["record_id"], prediction.record_id)


if __name__ == "__main__":
    unittest.main()
