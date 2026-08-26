import copy
import json
import unittest
from pathlib import Path

from harness_matsci.adaptive_evidence_dataset import AdaptiveEvidenceExample


DATA = Path(
    "research/evl_iclr/data/adaptive_evidence_trajectories_v2_structured_6000.jsonl"
)


class AdaptiveEvidenceDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = json.loads(DATA.read_text(encoding="utf-8").splitlines()[0])

    def test_visible_features_are_invariant_to_hidden_outcome(self):
        original = AdaptiveEvidenceExample(self.trajectory)
        mutated_payload = copy.deepcopy(self.trajectory)
        hidden = mutated_payload["hidden_outcome_for_evaluation_only"]
        hidden["success_label"] = 1 - int(hidden["success_label"])
        hidden["normalized_utility"] = 1.0 - float(hidden["normalized_utility"])
        hidden["base_gain"] = -99.0
        mutated = AdaptiveEvidenceExample(mutated_payload)
        for stage in ("base", "screening", "verification"):
            self.assertEqual(original.central_features(stage), mutated.central_features(stage))
            self.assertEqual(
                original.current_feature_views(stage),
                mutated.current_feature_views(stage),
            )

    def test_future_evidence_views_are_numeric_and_replicated(self):
        example = AdaptiveEvidenceExample(self.trajectory)
        screening = example.counterfactual_next_stage_views("base")
        verification = example.counterfactual_next_stage_views("screening")
        self.assertGreaterEqual(len(screening), 2)
        self.assertGreaterEqual(len(verification), 2)
        self.assertTrue(all(value == value for row in screening for value in row.values()))
        self.assertTrue(
            all("signal_screening__margin" in row for row in screening)
        )
        self.assertGreater(
            len({row["signal_screening__margin"] for row in screening}), 1
        )

    def test_realized_gain_charges_acquired_evidence_cost(self):
        example = AdaptiveEvidenceExample(self.trajectory)
        self.assertGreater(
            example.realized_gain("base"), example.realized_gain("verification")
        )


if __name__ == "__main__":
    unittest.main()
