import unittest

from experiments.sd_worth.evaluate_same_evidence_llm_judges import (
    evaluate_same_evidence_judges,
)


def payload(split, stage, count=30):
    rows = []
    for index in range(count):
        worthy = index % 3 == 0
        probability = 0.9 if worthy else 0.1
        target_gain = 0.8 if worthy else -0.2
        rows.append(
            {
                "record_id": f"{split}-{index}",
                "benchmark": "task",
                "split": split,
                "stage": stage,
                "acquired_evidence_cost": 0.1 if stage != "base" else 0.0,
                "target_gain": target_gain,
                "mean_p_positive_gain": probability,
                "mean_expected_gain": target_gain,
                "models": {
                    "model": {
                        "p_positive_gain": probability,
                        "expected_gain": target_gain,
                    }
                },
            }
        )
    return {"models": ["model"], "rows": rows}


class SameEvidenceLLMJudgesTest(unittest.TestCase):
    def test_evaluation_selects_positive_actions_and_charges_evidence(self):
        result = evaluate_same_evidence_judges(
            {"screening": payload("feedback", "screening")},
            {"screening": payload("acceptance", "screening")},
            budget_fraction=0.1,
            alpha=0.1,
        )
        strongest = result["strongest_result"]
        self.assertEqual(strongest["selected_count"], 3)
        self.assertEqual(len(strongest["selected_record_ids"]), 3)
        self.assertEqual(strongest["errors"], 0)
        self.assertLess(
            strongest["trajectory_total_strict_gain"],
            strongest["selected_total_strict_gain"],
        )


if __name__ == "__main__":
    unittest.main()
