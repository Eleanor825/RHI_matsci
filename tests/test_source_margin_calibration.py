import unittest

from harness_matsci.source_fusion import SourceFusionObservation
from harness_matsci.source_margin_calibration import fit_source_margin_calibrator


class SourceMarginCalibrationTests(unittest.TestCase):
    def test_monotonic_calibration_recovers_nonlinear_gain(self):
        observations = []
        for index in range(200):
            margin = -1.0 + 2.0 * index / 199.0
            gain = max(0.0, margin) - 0.2
            observations.append(
                SourceFusionObservation(
                    record_id=str(index),
                    benchmark="task",
                    source_names=("source",),
                    predicted_gain=((margin - 0.01,), (margin,), (margin + 0.01,)),
                    gain=gain,
                )
            )
        model = fit_source_margin_calibrator(observations)
        predictions = model.predict(observations)
        low = sum(value[0] for value in predictions[20].predicted_gain) / 3.0
        high = sum(value[0] for value in predictions[-1].predicted_gain) / 3.0
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
