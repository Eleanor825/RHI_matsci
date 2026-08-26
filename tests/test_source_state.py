import unittest

from harness_matsci.source_fusion import SourceFusionObservation
from harness_matsci.source_state import fit_source_state_model


class SourceStateTests(unittest.TestCase):
    def test_projected_gain_space_skips_feedback_margin_map(self):
        observations = []
        for index in range(160):
            gain = -0.6 + 1.2 * index / 159.0
            observations.append(
                SourceFusionObservation(
                    record_id=str(index),
                    benchmark="task",
                    source_names=("source_a", "source_b"),
                    predicted_gain=(
                        (gain - 0.03, gain - 0.01),
                        (gain, gain + 0.01),
                        (gain + 0.03, gain + 0.02),
                    ),
                    gain=gain,
                )
            )
        model = fit_source_state_model(
            observations,
            calibrate_source_margins=False,
        )
        self.assertIsNone(model.margin_calibrator)
        predictions = model.predict(observations)
        self.assertLess(predictions[0].mean_gain, predictions[-1].mean_gain)


if __name__ == "__main__":
    unittest.main()
