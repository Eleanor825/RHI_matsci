import unittest

from harness_matsci.external_h17_pairwise import _regime_quotas


class ExternalH17PairwiseTests(unittest.TestCase):
    def test_regime_quotas_cover_requested_pairs(self):
        quotas = _regime_quotas(101)
        self.assertEqual(sum(quotas.values()), 101)
        self.assertEqual(quotas["easy"], 40)
        self.assertEqual(quotas["ambiguous"], 30)
        self.assertEqual(quotas["ood"], 31)


if __name__ == "__main__":
    unittest.main()
