from __future__ import annotations

import unittest

from harness_matsci.source_decomposition import decompose_source_variance


class SourceDecompositionTests(unittest.TestCase):
    def test_law_of_total_variance_decomposition(self) -> None:
        result = decompose_source_variance([[0.0, 1.0], [2.0, 3.0]])
        self.assertAlmostEqual(result.internal_variance, 1.0)
        self.assertAlmostEqual(result.external_variance, 0.25)
        self.assertAlmostEqual(result.interaction_variance, 0.0)
        self.assertAlmostEqual(result.total_variance, 1.25)

    def test_separates_replica_source_interaction(self) -> None:
        result = decompose_source_variance([[0.0, 2.0], [2.0, 0.0]])
        self.assertAlmostEqual(result.internal_variance, 0.0)
        self.assertAlmostEqual(result.external_variance, 0.0)
        self.assertAlmostEqual(result.interaction_variance, 1.0)
        self.assertAlmostEqual(result.total_variance, 1.0)

    def test_requires_balanced_replicates(self) -> None:
        with self.assertRaises(ValueError):
            decompose_source_variance([[0.0, 1.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
