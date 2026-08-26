import unittest

from harness_matsci.sequential_tool_experiment import (
    _evaluate_threshold,
    _run_harness_evolution,
    _run_joint_certificate_aware_policy,
    _run_safe_activation_policy,
)
from harness_matsci.schema import ActionRecord


class SequentialToolExperimentTests(unittest.TestCase):
    def test_threshold_metrics_subtract_tool_cost(self):
        records = [
            ActionRecord(
                record_id=str(index),
                benchmark="matbench_pairwise",
                split="test",
                visible_context="x",
                candidate_action="a",
                action_type="choose_candidate",
                evidence=[],
                features={},
                label=label,
                metadata={"group_id": "g"},
            )
            for index, label in enumerate((1, 0))
        ]
        result = _evaluate_threshold(
            [0.9, 0.1],
            [1, 0],
            [0.4, -0.2],
            records,
            threshold=0.5,
            tool_cost=0.05,
            acquired_count=2,
        )
        self.assertAlmostEqual(result["coverage"], 0.5)
        self.assertAlmostEqual(result["selective_risk"], 0.0)
        self.assertAlmostEqual(result["selected_mean_gain_after_tool_cost"], 0.35)
        self.assertAlmostEqual(result["population_net_gain_all_actions_acquire_tool"], 0.15)

    def test_harness_evolution_requires_certified_gain_improvement(self):
        records = [
            ActionRecord(
                record_id=str(index),
                benchmark="matbench_pairwise",
                split=split,
                visible_context="x",
                candidate_action="a",
                action_type="choose_candidate",
                evidence=[],
                features={},
                label=int(index < 50),
                metadata={"group_id": "g"},
            )
            for split in ("acceptance", "test")
            for index in range(100)
        ]
        splits = {"acceptance": records[:100], "test": records[100:]}
        labels = {split: [record.label for record in rows] for split, rows in splits.items()}
        gains = {
            split: [0.4 if record.label else -0.8 for record in rows]
            for split, rows in splits.items()
        }
        losses = {
            split: [0.0 if record.label else 1.0 for record in rows]
            for split, rows in splits.items()
        }
        good_scores = [0.95 if index < 50 else 0.05 for index in range(100)]
        bad_scores = [0.5 for _ in range(100)]
        methods = {
            "base_mean": {"acceptance": bad_scores, "test": bad_scores},
            "sparse_verification_mean": {"acceptance": good_scores, "test": good_scores},
            "sparse_tool_probability": {"acceptance": bad_scores, "test": bad_scores},
            "task_sparse_continuous_mean": {"acceptance": good_scores, "test": good_scores},
            "task_sparse_continuous_sd_worth": {"acceptance": good_scores, "test": good_scores},
        }
        metadata = {
            method: {
                "primary_tool_cost": 0.002,
                "acquisition_counts": {"acceptance": 50, "test": 50},
            }
            for method in methods
        }
        metadata["base_mean"] = {
            "acquisition_counts": {"acceptance": 0, "test": 0}
        }
        evolution = _run_harness_evolution(
            methods,
            metadata,
            labels,
            gains,
            losses,
            splits,
            alpha=0.20,
            delta=0.05,
        )
        self.assertEqual(evolution["selected_harness"], "sparse_verification_mean")
        self.assertGreater(evolution["selected_result"]["test"]["population_net_gain"], 0.0)

    def test_harness_evolution_supports_hoeffding_shortfall_certificate(self):
        records = [
            ActionRecord(
                record_id=f"{split}-{index}",
                benchmark="matbench_pairwise",
                split=split,
                visible_context="x",
                candidate_action="a",
                action_type="choose_candidate",
                evidence=[],
                features={},
                label=int(index < 250),
                metadata={"group_id": "g"},
            )
            for split in ("acceptance", "test")
            for index in range(500)
        ]
        splits = {"acceptance": records[:500], "test": records[500:]}
        labels = {split: [record.label for record in rows] for split, rows in splits.items()}
        gains = {
            split: [0.4 if record.label else -0.8 for record in rows]
            for split, rows in splits.items()
        }
        losses = {
            split: [0.0 if record.label else 1.0 for record in rows]
            for split, rows in splits.items()
        }
        good_scores = [0.95 if index < 250 else 0.05 for index in range(500)]
        bad_scores = [0.5 for _ in range(500)]
        methods = {
            "base_mean": {"acceptance": bad_scores, "test": bad_scores},
            "sparse_verification_mean": {"acceptance": good_scores, "test": good_scores},
            "sparse_tool_probability": {"acceptance": bad_scores, "test": bad_scores},
            "task_sparse_continuous_mean": {"acceptance": good_scores, "test": good_scores},
            "task_sparse_continuous_sd_worth": {"acceptance": good_scores, "test": good_scores},
        }
        metadata = {
            method: {
                "primary_tool_cost": 0.002,
                "acquisition_counts": {"acceptance": 250, "test": 250},
            }
            for method in methods
        }
        metadata["base_mean"] = {"acquisition_counts": {"acceptance": 0, "test": 0}}
        evolution = _run_harness_evolution(
            methods,
            metadata,
            labels,
            gains,
            losses,
            splits,
            alpha=0.20,
            delta=0.05,
            risk_family="hoeffding_shortfall",
        )
        self.assertEqual(evolution["risk_family"], "hoeffding_shortfall")
        self.assertEqual(evolution["selected_harness"], "sparse_verification_mean")
        self.assertGreater(evolution["selected_result"]["test"]["population_net_gain"], 0.0)

    def test_joint_policy_acquires_only_for_certified_positive_gain(self):
        splits = self._joint_policy_records(1000)
        labels = {split: [record.label for record in rows] for split, rows in splits.items()}
        gains = {
            split: [0.5 if record.label else -0.5 for record in rows]
            for split, rows in splits.items()
        }
        losses = {
            split: [0.0 if record.label else 1.0 for record in rows]
            for split, rows in splits.items()
        }
        base = {
            split: [1.0 - index / 1000 for index in range(1000)]
            for split in splits
        }
        tool = {
            split: [0.99 if index < 100 else 0.01 for index in range(1000)]
            for split in splits
        }
        result = _run_joint_certificate_aware_policy(
            base, tool, labels, gains, losses, splits, alpha=0.20, delta=0.05, tool_cost=0.002
        )
        self.assertEqual(result["selected_policy"], "certified_acquire_then_execute")
        self.assertGreater(result["test"]["population_net_gain"], 0.0)
        self.assertGreater(result["test"]["coverage"], 0.0)

    def test_joint_policy_falls_back_to_zero_cost_no_op(self):
        splits = self._joint_policy_records(1000)
        labels = {split: [record.label for record in rows] for split, rows in splits.items()}
        gains = {split: [-0.5] * 1000 for split in splits}
        losses = {split: [1.0] * 1000 for split in splits}
        flat = {split: [0.5] * 1000 for split in splits}
        result = _run_joint_certificate_aware_policy(
            flat, flat, labels, gains, losses, splits, alpha=0.20, delta=0.05, tool_cost=0.002
        )
        self.assertEqual(result["selected_policy"], "no_op")
        self.assertEqual(result["test"]["acquired_count"], 0)
        self.assertEqual(result["test"]["population_net_gain"], 0.0)

    def test_safe_activation_preserves_frozen_contract_or_no_op(self):
        splits = self._joint_policy_records(1000)
        labels = {split: [record.label for record in rows] for split, rows in splits.items()}
        gains = {
            split: [0.5 if record.label else -0.5 for record in rows]
            for split, rows in splits.items()
        }
        losses = {
            split: [0.0 if record.label else 1.0 for record in rows]
            for split, rows in splits.items()
        }
        scores = {
            split: [0.99 if index < 100 else 0.01 for index in range(1000)]
            for split in splits
        }
        metadata = {
            "primary_tool_cost": 0.002,
            "acquisition_counts": {"acceptance": 100, "test": 100},
        }
        active = _run_safe_activation_policy(
            scores, metadata, labels, gains, losses, splits, alpha=0.20, delta=0.05
        )
        self.assertEqual(active["selected_policy"], "activate_frozen_contract")
        self.assertGreater(active["test"]["population_net_gain"], 0.0)
        bad_gains = {split: [-0.5] * 1000 for split in splits}
        inactive = _run_safe_activation_policy(
            scores, metadata, labels, bad_gains, losses, splits, alpha=0.20, delta=0.05
        )
        self.assertEqual(inactive["selected_policy"], "no_op")
        self.assertEqual(inactive["test"]["acquired_count"], 0)

    @staticmethod
    def _joint_policy_records(count):
        return {
            split: [
                ActionRecord(
                    record_id=f"{split}-{index}",
                    benchmark="matbench_pairwise",
                    split=split,
                    visible_context="x",
                    candidate_action="a",
                    action_type="choose_candidate",
                    evidence=[],
                    features={},
                    label=int(index < 100),
                    metadata={"group_id": "g"},
                )
                for index in range(count)
            ]
            for split in ("acceptance", "test")
        }


if __name__ == "__main__":
    unittest.main()
