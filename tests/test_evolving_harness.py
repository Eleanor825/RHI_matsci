import unittest

from harness_matsci.benchmarks import make_records
from harness_matsci.evolving_harness import fit_numeric_critic, run_evolving_harness, score_action_records
from harness_matsci.training import split_records, train_gate


class EvolvingHarnessTest(unittest.TestCase):
    def test_runtime_scores_are_numeric_and_oracle_free(self):
        records = make_records("preferential_bo", n=100, seed=13)
        train, validation, _ = split_records(records, seed=13)
        critic = fit_numeric_critic(
            train,
            validation,
            feature_names=("verbal_confidence", "evidence_support"),
            alpha=0.1,
            epochs=30,
        )
        scores = score_action_records(records[:5], critic)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(0.0 <= row["p_success"] <= 1.0 for row in scores))
        self.assertTrue(all(row["uncertainty"] >= 0.0 for row in scores))
        self.assertNotIn("label", scores[0])
        self.assertNotIn("utility", scores[0])
        self.assertIn("utility_if_success", scores[0])

    def test_evolution_keeps_test_hidden_until_final_report(self):
        records = make_records("preferential_bo", n=120, seed=4)
        report = run_evolving_harness(records, iterations=2, seed=4, epochs=30)
        self.assertEqual(report["sizes"], {"train": 72, "feedback": 18, "acceptance": 18, "test": 12})
        self.assertEqual(len(report["revisions"]), 2)
        self.assertEqual(len(report["action_scores"]), 12)
        self.assertTrue(report["method"]["test_is_used_only_after_selection"])


if __name__ == "__main__":
    unittest.main()
