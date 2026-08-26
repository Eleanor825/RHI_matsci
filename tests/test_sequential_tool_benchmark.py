import unittest
from pathlib import Path

from harness_matsci.historical import MAIN_MATERIAL_TASKS, grouped_four_way_split, load_historical_tasks
from harness_matsci.sequential_tool_benchmark import (
    RealScientificToolEnvironment,
    SequentialToolEnvironment,
    TrainOnlyStructuredSurrogateToolEnvironment,
    TrainOnlySurrogateToolEnvironment,
    audit_visible_payload,
)


RAW = Path("/Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks")


@unittest.skipUnless(RAW.exists(), "historical material-discovery data are not available")
class SequentialToolBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records_by_task = load_historical_tasks(RAW, MAIN_MATERIAL_TASKS)
        cls.splits = {task: grouped_four_way_split(records, seed=7) for task, records in records_by_task.items()}
        train = [record for task in MAIN_MATERIAL_TASKS for record in cls.splits[task]["train"]]
        cls.environment = SequentialToolEnvironment.fit(RAW, train)
        cls.surrogate_environment = TrainOnlySurrogateToolEnvironment.fit(
            RAW, train, environment_seed=7
        )
        cls.structured_surrogate_environment = TrainOnlyStructuredSurrogateToolEnvironment.fit(
            RAW, train, environment_seed=7
        )
        cls.real_tool_environment = RealScientificToolEnvironment.fit(
            RAW, train, environment_seed=7
        )

    def test_visible_observations_pass_leakage_audit(self):
        for task in MAIN_MATERIAL_TASKS:
            record = self.splits[task]["test"][0]
            for stage in ("base", "screening", "verification"):
                observation = self.environment.observe(record, stage=stage, evidence_replica=0, run_seed=7)
                audit_visible_payload(observation.visible_payload())
                self.assertNotIn("audit_only", observation.to_json())

    def test_all_tasks_have_high_label_margin_consistency(self):
        records = [
            record
            for task in MAIN_MATERIAL_TASKS
            for split in self.splits[task].values()
            for record in split
        ]
        consistency = self.environment.label_consistency(records)
        for task, value in consistency.items():
            self.assertGreaterEqual(value, 0.99, task)

    def test_verification_is_more_precise_than_screening(self):
        for task in MAIN_MATERIAL_TASKS:
            record = self.splits[task]["test"][0]
            latent = self.environment.latent_margin(record)
            screening_errors = []
            verification_errors = []
            for replica in range(50):
                screening = self.environment.observe(
                    record, stage="screening", evidence_replica=replica, run_seed=11
                )
                verification = self.environment.observe(
                    record, stage="verification", evidence_replica=replica, run_seed=11
                )
                screening_errors.append(abs(screening.record.features["tool_estimated_margin"] - latent))
                verification_errors.append(abs(verification.record.features["tool_estimated_margin"] - latent))
            self.assertLess(sum(verification_errors), sum(screening_errors), task)

    def test_train_only_surrogate_is_invariant_to_nontraining_hidden_outcomes(self):
        test_records = [self.splits[task]["test"][0] for task in MAIN_MATERIAL_TASKS]
        for environment in (
            self.surrogate_environment,
            self.structured_surrogate_environment,
            self.real_tool_environment,
        ):
            audit = environment.audit_nontraining_outcome_invariance(
                test_records,
                evidence_replica=101,
                run_seed=1007,
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["audited_records"], 3)
            for record in test_records:
                observation = environment.observe(
                    record,
                    stage="verification",
                    evidence_replica=101,
                    run_seed=1007,
                )
                self.assertEqual(
                    observation.record.metadata["tool_observation_semantics"],
                    "train_only_surrogate_prediction",
                )
                self.assertFalse(
                    observation.provenance["nontraining_outcome_used_for_observation"]
                )

    def test_real_scientific_tools_emit_domain_outputs(self):
        expected = {
            "matbench_pairwise": "predicted_chosen_normalized_bulk_modulus",
            "discover_unique": "train_library_novelty_percentile",
            "extreme_properties": "rdkit_generated_properties",
        }
        for task in MAIN_MATERIAL_TASKS:
            record = self.splits[task]["test"][0]
            observation = self.real_tool_environment.observe(
                record,
                stage="verification",
                evidence_replica=101,
                run_seed=1007,
            )
            self.assertIn(expected[task], observation.visible_outputs)
            self.assertEqual(
                observation.provenance["observation_channel"],
                "matminer_rdkit_train_only_models",
            )


if __name__ == "__main__":
    unittest.main()
