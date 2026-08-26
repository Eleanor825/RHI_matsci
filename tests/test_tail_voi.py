import unittest

from harness_matsci.evidence_value import EvidenceValueObservation
from harness_matsci.tail_voi import fit_tail_voi_model


class TailVoITests(unittest.TestCase):
    def test_tail_model_prefers_high_positive_probability_low_downside(self):
        rows = []
        for index in range(200):
            signal = (index % 20) / 20.0
            rows.append(
                EvidenceValueObservation(
                    record_id=str(index),
                    benchmark="task",
                    base_mean_gain=signal,
                    base_p_positive_gain=signal,
                    base_expected_shortfall=1.0 - signal,
                    base_tail_risk_score=signal - 0.5,
                    base_internal_variance=0.01,
                    base_external_variance=0.02,
                    tool_cost=0.02,
                    realized_value=0.2 if signal > 0.65 else -0.05,
                )
            )
        model = fit_tail_voi_model(rows, seed=5, replicas=3)
        low = model.predict_one(rows[1])
        high = model.predict_one(rows[19])
        self.assertGreater(high.positive_probability, low.positive_probability)
        self.assertGreater(high.tail_score, low.tail_score)
        self.assertGreaterEqual(high.epistemic_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
