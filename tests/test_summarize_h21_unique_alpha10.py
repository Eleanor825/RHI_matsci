import unittest

from experiments.sd_worth.summarize_h21_unique_alpha10 import summarize


def _report(primary_gain, no_source_gain, *, certified=True, errors=0):
    def method(gain, deployed, method_errors):
        return {
            "deployed": deployed,
            "deployed_test_population_gain": gain if deployed else 0.0,
            "test_evaluation": {
                "executed_count": 100,
                "errors": method_errors,
                "selective_risk": method_errors / 100,
                "execution_coverage": 0.10,
                "acquired_count": 100,
                "acquisition_coverage": 0.10,
                "population_gain_after_cost": gain,
            },
            "acceptance_certificate": {
                "executed_count": 100,
                "errors": method_errors,
                "population_gain_after_cost": gain,
            },
        }

    return {
        "policies": {
            "source_voi_expected": method(primary_gain, certified, errors),
            "source_voi_probability": method(primary_gain - 0.0001, certified, errors),
            "no_source_voi_expected": method(no_source_gain, certified, errors),
            "external_variance": method(0.0, False, 0),
        }
    }


class SummarizeH21Tests(unittest.TestCase):
    def test_confirmatory_gate_uses_frozen_primary(self):
        reports = [_report(0.004, 0.003) for _ in range(4)]
        summary = summarize(reports, folds=[1, 2, 3, 4])
        self.assertTrue(summary["confirmatory_gate"]["passed"])
        self.assertAlmostEqual(summary["source_ablation_gain_delta"], 0.001)
        self.assertEqual(
            summary["evolved_contract"]["folds"][0]["selected_contract"],
            "source_voi_expected",
        )

    def test_uncertified_fold_contributes_no_execution_or_gain(self):
        reports = [_report(0.004, 0.003) for _ in range(3)] + [
            _report(0.004, 0.003, certified=False)
        ]
        summary = summarize(reports, folds=[1, 2, 3, 4])
        primary = summary["methods"]["source_voi_expected"]
        self.assertEqual(primary["certified_fold_count"], 3)
        self.assertEqual(primary["pooled_executed_count"], 300)


if __name__ == "__main__":
    unittest.main()
