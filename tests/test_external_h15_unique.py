import unittest
from pathlib import Path

from harness_matsci.external_h15_unique import build_external_h15_unique_fold


class ExternalH15UniqueTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("benchmarks/external_h15_unique/raw/matbench_log_kvrh.json.bz2").exists(),
        "H15 unique snapshot unavailable",
    )
    def test_fold_is_group_disjoint_and_powered(self):
        fold = build_external_h15_unique_fold(
            "benchmarks/external_h15_unique/raw/matbench_log_kvrh.json.bz2",
            fold=0,
        )
        groups = {split: set(values) for split, values in fold.groups_by_split.items()}
        self.assertFalse(groups["train"] & groups["test"])
        self.assertGreater(len(fold.records_by_split["acceptance"]), 1000)
        self.assertGreater(
            sum(record.label for record in fold.records_by_split["acceptance"]),
            50,
        )


if __name__ == "__main__":
    unittest.main()
