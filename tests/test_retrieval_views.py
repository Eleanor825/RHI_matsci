from __future__ import annotations

import unittest

from harness_matsci.retrieval_views import HistoricalEvidenceRetriever
from harness_matsci.schema import ActionRecord


def record(record_id: str, label: int, value: float) -> ActionRecord:
    return ActionRecord(
        record_id=record_id,
        benchmark="matbench_pairwise",
        split="train",
        visible_context=f"context {record_id}",
        candidate_action=f"action {record_id}",
        action_type="choose_candidate",
        evidence=["base evidence"],
        features={"candidate_domain_position": value, "evidence_support": 0.5},
        label=label,
        utility=value,
        metadata={"raw_record_id": record_id},
    )


class RetrievalViewTests(unittest.TestCase):
    def test_views_use_only_other_training_records(self) -> None:
        retriever = HistoricalEvidenceRetriever(
            [record("a", 1, 0.1), record("b", 0, 0.2), record("c", 1, 0.9)],
            neighbors=2,
        )
        target = record("target", 0, 0.15)
        views = retriever.evidence_views(target)
        self.assertEqual([view.view_id for view in views], [
            "tool::base_evidence", "tool::nearest_analogs", "tool::task_statistics"
        ])
        result_ids = [row["record_id"] for row in views[1].provenance["results"]]
        self.assertNotIn("target", result_ids)
        self.assertEqual(len(result_ids), 2)
        self.assertIn("historical_success=", "\n".join(views[1].record.evidence))

    def test_rejects_target_inside_library(self) -> None:
        target = record("a", 1, 0.1)
        retriever = HistoricalEvidenceRetriever([target, record("b", 0, 0.2)])
        with self.assertRaises(ValueError):
            retriever.evidence_views(target)


if __name__ == "__main__":
    unittest.main()
