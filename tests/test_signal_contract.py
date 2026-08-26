from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from harness_matsci.benchmarks import make_records
from harness_matsci.signal_contract import ALL_UNCERTAINTY_SIGNALS, require_signal_contract, signal_coverage
from harness_matsci.uncertainty_signals import AnswerLogitSignalProvider, attach_signal_features


class SignalContractTests(unittest.TestCase):
    def test_answer_logit_provider_attaches_internal_features(self) -> None:
        records = make_records("preferential_bo", n=2, seed=3)

        class FakeJudge:
            def score(self, prompts):
                return [
                    SimpleNamespace(
                        margin=1.5,
                        confidence=0.8175744762,
                        uncertainty=0.1824255238,
                        preferred="A",
                    )
                    for _ in prompts
                ]

        augmented = attach_signal_features(records, AnswerLogitSignalProvider(FakeJudge()))
        for record in augmented:
            self.assertAlmostEqual(record.features["answer_logit_margin"], 1.5)
            self.assertGreater(record.features["answer_logit_execute_probability"], 0.8)
            self.assertAlmostEqual(record.features["answer_logit_uncertainty"], 0.1824255238)

    def test_strict_contract_rejects_silent_missing_signals(self) -> None:
        records = make_records("preferential_bo", n=2, seed=4)
        with self.assertRaises(ValueError):
            require_signal_contract(records)

    def test_complete_contract_passes_and_reports_full_coverage(self) -> None:
        records = make_records("preferential_bo", n=2, seed=5)
        completed = [
            replace(
                record,
                features={**record.features, **{name: 0.5 for name in ALL_UNCERTAINTY_SIGNALS}},
            )
            for record in records
        ]
        require_signal_contract(completed)
        coverage = signal_coverage(completed)
        self.assertTrue(all(item["fraction"] == 1.0 for item in coverage.values()))


if __name__ == "__main__":
    unittest.main()
