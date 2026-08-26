import csv
import tempfile
import unittest
from pathlib import Path

from harness_matsci.matbot_lfp_historical import (
    build_matbot_lfp_historical_choices,
    build_matbot_lfp_historical_trajectories,
    load_matbot_lfp_historical_actions,
)


class MatBotLfpHistoricalTests(unittest.TestCase):
    def test_hides_outcomes_from_visible_context(self):
        fields = [
            "source_doc",
            "source_batch",
            "pair_id",
            "weighted_EC",
            "weighted_LiPF6",
            "全配方电导率/mS/cm_num",
            "-20℃ DCR（折算）_num",
            "-15℃ DCR（50%SOC，1C放电10s）_num",
            "45℃循环600圈_num",
        ]
        rows = [
            ["doc", "batch_1_legacy_minus20_202604", "A+B", "30", "12", "14", "300", "", "92"],
            ["doc", "batch_1_legacy_minus20_202604", "C+D", "28", "13", "13", "350", "", "90"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            records = load_matbot_lfp_historical_actions(path)
        self.assertEqual(len(records), 2)
        self.assertNotIn("300", records[0].visible_context)
        self.assertNotIn("92", records[0].visible_context)
        self.assertEqual(
            records[0].metadata["hidden_outcome_for_evaluation_only"]["primary_dcr"],
            300.0,
        )
        self.assertEqual({record.label for record in records}, {0, 1})
        trajectories = build_matbot_lfp_historical_trajectories(records)
        self.assertTrue(all(trajectory.complete for trajectory in trajectories))
        self.assertTrue(all(trajectory.steps[0].observed_outcome for trajectory in trajectories))
        self.assertTrue(all(not trajectory.steps[0].internal_signals for trajectory in trajectories))
        choices = build_matbot_lfp_historical_choices(records)
        self.assertEqual(len(choices), 1)
        self.assertNotIn("primary_dcr", choices[0].visible_context)
        self.assertFalse(choices[0].metadata["orientation_uses_hidden_outcomes"])
        self.assertIn(choices[0].label, (0, 1))
        self.assertTrue(
            any(
                name.startswith("candidate_delta::component_fraction::")
                for name in choices[0].features
            )
        )


if __name__ == "__main__":
    unittest.main()
