import unittest

from harness_matsci.conditional_action_rhi import _stable_failure_patterns, add_policy_features, feature_policy
from harness_matsci.conditional_action_rhi import _fixed_budget_report
from harness_matsci.schema import ActionRecord
from harness_matsci.graph_rhi import BRANCHES


def _record(record_id: str, task: str, stage: str, label: int = 1) -> ActionRecord:
    return ActionRecord(
        record_id=record_id,
        benchmark=task,
        split="feedback",
        visible_context="visible context",
        candidate_action="candidate action",
        action_type="choose_candidate",
        evidence=["evidence"],
        features={
            "evidence_support": 0.8,
            "verbal_confidence": 0.7,
            "cost": 0.2,
            "reversibility": 0.8,
            "tool_agreement": 0.7,
            "ood_score": 0.1,
        },
        label=label,
        utility=0.5 if label else 0.0,
        metadata={"trajectory_id": f"traj-{record_id}", "task": task, "stage": stage, "action_index": 0, "action_count": 1},
    )


class ConditionalActionRHITests(unittest.TestCase):
    def test_conditioned_policy_contains_task_and_stage_features(self):
        names = feature_policy("extreme_property_discovery", "commit", {"evidence_support"})
        self.assertIn("evidence_support__task__extreme_property_discovery", names)
        self.assertIn("evidence_support__stage__commit", names)

    def test_policy_features_do_not_include_outcome_fields(self):
        record = _record("a", "materials_pairwise_preference", "proposal")
        prepared = add_policy_features([record], {"evidence_support"})[0]
        self.assertNotIn("label", prepared.features)
        self.assertNotIn("utility", prepared.features)
        self.assertEqual(prepared.features["is_commit"], 0.0)

    def test_single_small_group_does_not_trigger_stable_mutation(self):
        records = [_record("a", "materials_pairwise_preference", "commit", label=0)]
        patterns = _stable_failure_patterns(records, [0.9], threshold=0.7, min_group=20)
        self.assertEqual(patterns, [])

    def test_fixed_budget_report_selects_exact_top_k(self):
        report = _fixed_budget_report([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], [0.9, 0.8, 0.7, 0.1], 0.5)
        self.assertEqual(report["budget"], 2.0)
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["risk"], 0.5)

    def test_graph_has_two_disjoint_mutation_branches(self):
        self.assertEqual(set(BRANCHES), {"H1_evidence_source", "H2_ood_stability"})
        self.assertTrue(set(BRANCHES["H1_evidence_source"]).isdisjoint(BRANCHES["H2_ood_stability"]))
