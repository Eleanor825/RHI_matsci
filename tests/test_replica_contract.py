from __future__ import annotations

import unittest

from harness_matsci.replica_contract import (
    ActionReplicaGrid,
    AgentEvidencePrediction,
    EXTERNAL_VIEW_FEATURES,
    controlled_evidence_views,
)
from harness_matsci.schema import ActionRecord


class ReplicaContractTests(unittest.TestCase):
    def _prediction(self, agent: str, view: str, gain: float) -> AgentEvidencePrediction:
        return AgentEvidencePrediction(
            record_id="r1",
            agent_replica_id=agent,
            evidence_view_id=view,
            predicted_gain=gain,
            positive_gain_probability=0.5,
            backend="test",
            internal_signals={},
            external_signals={},
        )

    def test_balanced_grid_summary(self) -> None:
        grid = ActionReplicaGrid.from_predictions(
            [
                self._prediction("a1", "e1", 0.0),
                self._prediction("a1", "e2", 1.0),
                self._prediction("a2", "e1", 2.0),
                self._prediction("a2", "e2", 3.0),
            ]
        )
        summary = grid.summarize()
        self.assertAlmostEqual(summary.mean_predicted_gain, 1.5)
        self.assertAlmostEqual(summary.source_variance.internal_variance, 1.0)
        self.assertAlmostEqual(summary.source_variance.external_variance, 0.25)

    def test_rejects_missing_cell(self) -> None:
        with self.assertRaises(ValueError):
            ActionReplicaGrid.from_predictions(
                [
                    self._prediction("a1", "e1", 0.0),
                    self._prediction("a1", "e2", 1.0),
                    self._prediction("a2", "e1", 2.0),
                ]
            )

    def test_evidence_views_do_not_change_internal_features(self) -> None:
        record = ActionRecord(
            record_id="r1",
            benchmark="matbench_pairwise",
            split="test",
            visible_context="context",
            candidate_action="choose candidate",
            action_type="choose_candidate",
            evidence=["source one", "source two"],
            features={
                "verbal_confidence": 0.7,
                "perturbation_stability": 0.8,
                "evidence_support": 0.9,
                "evidence_conflict": 0.0,
                "source_reliability": 0.9,
                "ood_score": 0.1,
                "evidence_count": 1.0,
            },
            label=1,
        )
        views = controlled_evidence_views(record)
        self.assertEqual(len(views), 3)
        for view in views:
            for key, value in record.features.items():
                if key not in EXTERNAL_VIEW_FEATURES:
                    self.assertEqual(view.record.features[key], value)
        self.assertGreater(views[2].record.features["evidence_conflict"], 0.0)


if __name__ == "__main__":
    unittest.main()
