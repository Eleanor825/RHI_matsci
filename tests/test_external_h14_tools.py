import json
import unittest
from pathlib import Path

from harness_matsci.external_h14_tools import ExternalH14EvidenceEnvironment
from harness_matsci.schema import ActionRecord


class ExternalH14ToolTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("benchmarks/external_h14/folds/fold_0/manifest.json").exists(),
        "external H14 snapshot is unavailable",
    )
    def test_views_are_numeric_and_do_not_expose_outcome(self):
        root = Path("benchmarks/external_h14/folds/fold_0")
        train = []
        for path in root.glob("*.train.jsonl"):
            train.extend(
                ActionRecord.from_json(json.loads(line))
                for line in path.read_text().splitlines()
                if line.strip()
            )
        test_path = root / "external_unique_jdft2d.test.jsonl"
        record = ActionRecord.from_json(json.loads(test_path.read_text().splitlines()[0]))
        environment = ExternalH14EvidenceEnvironment.fit(
            "benchmarks/external_h14/raw", train, seed=1401
        )
        views = environment.evidence_views(record)
        self.assertEqual(len(views), 3)
        self.assertEqual(
            {view.metadata["evidence_source"] for view in views},
            {"composition", "structure", "high_fidelity"},
        )
        for view in views:
            self.assertIn("tool_estimated_margin", view.features)
            self.assertIn("tool_estimated_utility", view.features)
            self.assertIn("tool_estimated_utility_low", view.features)
            self.assertIn("tool_estimated_utility_high", view.features)
            self.assertGreaterEqual(view.features["tool_estimated_utility"], 0.0)
            self.assertLessEqual(view.features["tool_estimated_utility"], 1.0)
            self.assertFalse(view.metadata["hidden_outcome_exposed"])
        high_fidelity = next(
            view for view in views if view.metadata["evidence_source"] == "high_fidelity"
        )
        self.assertTrue(high_fidelity.metadata["outcome_defining_measurement_replayed"])
        self.assertEqual(high_fidelity.features["tool_cost"], 0.020)


if __name__ == "__main__":
    unittest.main()
