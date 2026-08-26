import unittest

from harness_matsci.contract_evolution import ContractCandidate, evolve_contract


class ContractEvolutionTests(unittest.TestCase):
    def test_rejects_high_gain_contract_that_is_not_robustly_certified(self):
        unsafe = ContractCandidate(
            "joint", "joint", "expected", (True, False), (0.2, 0.0)
        )
        robust = ContractCandidate(
            "verify", "verify", "expected", (True, True), (0.1, 0.1)
        )
        result = evolve_contract([unsafe, robust])
        self.assertEqual(result.selected.name, "verify")
        self.assertFalse(result.trace[0]["feasible"])
        self.assertTrue(result.trace[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
