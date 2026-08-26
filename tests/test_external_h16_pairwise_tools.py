import json
import unittest
from pathlib import Path

from harness_matsci.external_h16_pairwise_tools import (
    ExternalH16PairwiseEvidenceEnvironment,
)
from harness_matsci.schema import ActionRecord


class ExternalH16PairwiseToolTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("benchmarks/external_h16_pairwise/folds/fold_0/manifest.json").exists(),
        "H16 pairwise fold unavailable",
    )
    def test_pairwise_views_are_batched_and_outcome_hidden(self):
        root = Path("benchmarks/external_h16_pairwise/folds/fold_0")
        train = [
            ActionRecord.from_json(json.loads(line))
            for line in (root / "external_pairwise_log_kvrh.train.jsonl").read_text().splitlines()
            if line
        ]
        test = [
            ActionRecord.from_json(json.loads(line))
            for line in (root / "external_pairwise_log_kvrh.test.jsonl").read_text().splitlines()[:2]
            if line
        ]
        environment = ExternalH16PairwiseEvidenceEnvironment.fit(
            "benchmarks/external_h15_unique/raw/matbench_log_kvrh.json.bz2",
            train,
            seed=1601,
        )
        views = environment.observe_many(test, source="composition")
        self.assertEqual(len(views), 2)
        self.assertFalse(views[0].metadata["hidden_outcome_exposed"])
        self.assertGreaterEqual(views[0].features["tool_estimated_utility"], 0.0)


if __name__ == "__main__":
    unittest.main()
