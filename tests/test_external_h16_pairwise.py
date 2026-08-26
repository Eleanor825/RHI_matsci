import unittest
from pathlib import Path

from harness_matsci.external_h16_pairwise import build_external_h16_pairwise_fold


class ExternalH16PairwiseTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("benchmarks/external_h15_unique/raw/matbench_log_kvrh.json.bz2").exists(),
        "log-kvrh snapshot unavailable",
    )
    def test_pairwise_fold_is_material_group_disjoint_and_powered(self):
        fold = build_external_h16_pairwise_fold(
            "benchmarks/external_h15_unique/raw/matbench_log_kvrh.json.bz2",
            fold=0,
        )
        self.assertGreater(len(fold.records_by_split["acceptance"]), 1000)
        self.assertGreater(sum(row.label for row in fold.records_by_split["acceptance"]), 100)


if __name__ == "__main__":
    unittest.main()
