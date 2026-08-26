from __future__ import annotations

import unittest

from harness_matsci.source_calibration import (
    ReplicaWorthinessObservation,
    SourceCalibrationParameters,
    SourceDecomposedWorthinessCalibrator,
    fit_source_decomposed_calibrator,
)


class SourceCalibrationTests(unittest.TestCase):
    def test_positive_gain_probability_moves_toward_half_with_uncertainty(self) -> None:
        calibrator = SourceDecomposedWorthinessCalibrator({
            "task": SourceCalibrationParameters(
                bias=0.0, base_variance=0.01, internal_weight=1.0, external_weight=1.0
            )
        })
        certain = ReplicaWorthinessObservation("a", "task", 0.5, 0.0, 0.0, 1)
        uncertain = ReplicaWorthinessObservation("b", "task", 0.5, 1.0, 1.0, 1)
        self.assertGreater(calibrator.predict_one(certain), calibrator.predict_one(uncertain))
        self.assertGreater(calibrator.predict_one(uncertain), 0.5)

    def test_fits_task_conditional_parameters_and_outputs_scalar(self) -> None:
        observations = []
        for task in ("a", "b"):
            observations.extend([
                ReplicaWorthinessObservation(f"{task}-1", task, 0.8, 0.01, 0.01, 1),
                ReplicaWorthinessObservation(f"{task}-2", task, 0.5, 0.02, 0.02, 1),
                ReplicaWorthinessObservation(f"{task}-3", task, -0.4, 0.01, 0.01, 0),
                ReplicaWorthinessObservation(f"{task}-4", task, -0.8, 0.02, 0.02, 0),
            ])
        calibrator = fit_source_decomposed_calibrator(
            observations,
            bias_grid=(0.0,),
            base_standard_deviation_grid=(0.1, 0.4),
            source_weight_grid=(0.0, 1.0),
        )
        probabilities = calibrator.predict(observations)
        self.assertEqual(len(probabilities), len(observations))
        self.assertTrue(all(0.0 < value < 1.0 for value in probabilities))
        self.assertGreater(probabilities[0], probabilities[2])
        self.assertEqual(set(calibrator.parameters_by_task), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
