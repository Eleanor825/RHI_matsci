import unittest

from experiments.sd_worth.audit_h5_round2_acceptance import (
    BASELINES,
    PRIMARY,
    audit_acceptance,
)


class H5Round2AcceptanceAuditTest(unittest.TestCase):
    def test_non_llm_pass_does_not_unlock_test_without_llm_gate(self):
        acceptance = {
            PRIMARY: {
                "selected_count": 500,
                "errors": 4,
                "risk_upper_bound": 0.08,
                "selective_risk": 0.04,
                "trajectory_total_strict_gain": 12.0,
            }
        }
        for index, baseline in enumerate(BASELINES):
            acceptance[baseline] = {
                "trajectory_total_strict_gain": 5.0 + index
            }
        audit = audit_acceptance({"summary": {"acceptance": acceptance}})
        self.assertTrue(audit["non_llm_acceptance_gate_passed"])
        self.assertLessEqual(
            audit["outcome_blind_hoeffding_risk_upper_bound"], 0.1
        )
        self.assertFalse(audit["full_acceptance_gate_passed"])
        self.assertFalse(audit["test_unlocked"])


if __name__ == "__main__":
    unittest.main()
