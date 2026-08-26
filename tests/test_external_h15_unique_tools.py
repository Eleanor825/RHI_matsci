import json
import unittest
from pathlib import Path

from harness_matsci.external_h15_unique_tools import (
    ExternalH15UniqueEvidenceEnvironment,
)
from harness_matsci.schema import ActionRecord


class ExternalH15UniqueToolTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("benchmarks/external_h15_unique/folds/fold_0/manifest.json").exists(),
        "H15 unique fold unavailable",
    )
    def test_views_are_train_only_numeric_and_outcome_hidden(self):
        root = Path("benchmarks/external_h15_unique/folds/fold_0")
        train = [
            ActionRecord.from_json(json.loads(line))
            for line in (root / "external_unique_log_kvrh.train.jsonl").read_text().splitlines()
            if line
        ]
        record = ActionRecord.from_json(
            json.loads(
                (root / "external_unique_log_kvrh.test.jsonl").read_text().splitlines()[0]
            )
        )
        environment = ExternalH15UniqueEvidenceEnvironment.fit(
            "benchmarks/external_h15_unique/raw/matbench_log_kvrh.json.bz2",
            train,
            seed=1501,
        )
        views = [environment.observe(record, source=source) for source in ("composition", "structure", "high_fidelity")]
        self.assertEqual(len(views), 3)
        for view in views:
            self.assertGreaterEqual(view.features["tool_estimated_utility"], 0.0)
            self.assertLessEqual(view.features["tool_estimated_utility"], 1.0)
            self.assertFalse(view.metadata["hidden_outcome_exposed"])
        batch = environment.observe_many([record, record], source="composition")
        self.assertEqual(len(batch), 2)
        self.assertEqual(
            batch[0].features["tool_estimated_utility"],
            views[0].features["tool_estimated_utility"],
        )


if __name__ == "__main__":
    unittest.main()
