import unittest

from harness_matsci.hurdle_gain_calibration import (
    HurdleGainObservation,
    fit_hurdle_gain_calibrator,
)


class HurdleGainCalibrationTests(unittest.TestCase):
    def test_asymmetric_failures_reduce_tail_risk_score(self):
        observations = []
        for index in range(200):
            positive = index % 5 != 0
            observations.append(
                HurdleGainObservation(
                    record_id=str(index),
                    benchmark="task",
                    mean_gain=0.2,
                    internal_variance=0.01,
                    external_variance=0.02,
                    gain=0.05 if positive else -1.0,
                )
            )
        model = fit_hurdle_gain_calibrator(
            observations,
            beta=0.10,
            tail_weight=0.50,
        )
        prediction = model.predict_one(observations[0])
        self.assertGreater(prediction.p_positive_gain, 0.70)
        self.assertLess(prediction.mean_gain, 0.0)
        self.assertGreater(prediction.expected_shortfall, 0.5)
        self.assertLess(prediction.tail_risk_score, prediction.mean_gain)

    def test_source_variance_is_reported_separately(self):
        observations = []
        for index in range(160):
            mean_gain = -0.4 + 0.8 * index / 159.0
            external = (index % 8) / 20.0
            sign = 1.0 if mean_gain + (0.1 if index % 2 else -0.1) > 0 else -1.0
            gain = 0.02 + 0.2 * abs(mean_gain) if sign > 0 else -0.2 - external
            observations.append(
                HurdleGainObservation(
                    record_id=str(index),
                    benchmark="task",
                    mean_gain=mean_gain,
                    internal_variance=0.01 * (index % 3),
                    external_variance=external * external,
                    gain=gain,
                )
            )
        model = fit_hurdle_gain_calibrator(observations, use_source_variance=True)
        prediction = model.predict_one(observations[-1])
        self.assertGreaterEqual(prediction.calibrated_internal_variance, 0.0)
        self.assertGreaterEqual(prediction.calibrated_external_variance, 0.0)
        self.assertAlmostEqual(
            prediction.calibrated_variance,
            prediction.irreducible_variance
            + prediction.calibrated_internal_variance
            + prediction.calibrated_external_variance,
        )
        self.assertEqual(prediction.to_json()["record_id"], prediction.record_id)

    def test_no_source_ablation_zeroes_source_terms(self):
        observations = [
            HurdleGainObservation(
                record_id=str(index),
                benchmark="task",
                mean_gain=(index - 50) / 100.0,
                internal_variance=0.2,
                external_variance=0.3,
                gain=0.1 if index >= 50 else -0.4,
            )
            for index in range(100)
        ]
        model = fit_hurdle_gain_calibrator(observations, use_source_variance=False)
        prediction = model.predict_one(observations[-1])
        self.assertEqual(prediction.calibrated_internal_variance, 0.0)
        self.assertEqual(prediction.calibrated_external_variance, 0.0)

    def test_known_failure_gain_is_used_structurally(self):
        observations = []
        for index in range(200):
            observations.append(
                HurdleGainObservation(
                    record_id=str(index),
                    benchmark="task",
                    mean_gain=0.3,
                    internal_variance=0.0,
                    external_variance=0.0,
                    gain=0.05 if index % 5 else -1.0,
                    known_failure_gain=-1.0,
                )
            )
        model = fit_hurdle_gain_calibrator(observations)
        prediction = model.predict_one(observations[0])
        self.assertAlmostEqual(prediction.negative_component_mean, -1.0)
        self.assertLess(prediction.mean_gain, 0.0)

    def test_decision_instability_can_expand_calibrated_uncertainty(self):
        observations = []
        for index in range(200):
            unstable = index % 2 == 0
            observations.append(
                HurdleGainObservation(
                    record_id=str(index),
                    benchmark="task",
                    mean_gain=0.1 if not unstable else 0.0,
                    internal_variance=0.01,
                    external_variance=0.01,
                    internal_boundary_instability=1.0 if unstable else 0.0,
                    external_boundary_instability=0.5 if unstable else 0.0,
                    gain=(-0.5 if unstable else 0.5),
                )
            )
        model = fit_hurdle_gain_calibrator(
            observations,
            use_decision_instability=True,
        )
        unstable_prediction = model.predict_one(observations[0])
        stable_prediction = model.predict_one(observations[1])
        self.assertTrue(model.uses_decision_instability)
        self.assertGreater(
            unstable_prediction.calibrated_internal_boundary_instability
            + unstable_prediction.calibrated_external_boundary_instability,
            stable_prediction.calibrated_internal_boundary_instability
            + stable_prediction.calibrated_external_boundary_instability,
        )


if __name__ == "__main__":
    unittest.main()
