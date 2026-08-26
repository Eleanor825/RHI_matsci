from __future__ import annotations

import unittest

from harness_matsci.risk_control import (
    audit_risk_controlled_grid,
    binomial_upper_confidence,
    empirical_bernstein_lower_confidence,
    empirical_bernstein_upper_confidence,
    hoeffding_upper_confidence,
    select_bounded_risk_grid,
    select_hoeffding_risk_grid,
    select_risk_controlled_grid,
    select_risk_controlled_threshold,
)


class RiskControlTests(unittest.TestCase):
    def test_exact_upper_bound_is_conservative(self) -> None:
        upper = binomial_upper_confidence(0, 100, 0.05)
        self.assertGreater(upper, 0.02)
        self.assertLess(upper, 0.04)

    def test_selection_rejects_high_risk_prefixes(self) -> None:
        probabilities = [0.99 - index * 0.005 for index in range(100)]
        labels = [1] * 80 + [0] * 20
        selection = select_risk_controlled_threshold(probabilities, labels, alpha=0.1, delta=0.05)
        self.assertLessEqual(selection.risk_upper_bound, 0.1)
        self.assertLessEqual(selection.selected_count, 80)

    def test_predeclared_grid_controls_candidate_multiplicity(self) -> None:
        probabilities = [1.0 - index / 200.0 for index in range(100)]
        labels = [1] * 50 + [0] * 50
        selection = select_risk_controlled_grid(
            probabilities,
            labels,
            coverages=(0.1, 0.2, 0.5, 1.0),
            alpha=0.2,
            delta=0.05,
        )
        self.assertEqual(selection.candidate_prefixes, 4)
        self.assertLessEqual(selection.risk_upper_bound, 0.2)

    def test_grid_audit_exposes_error_and_confidence_bottlenecks(self) -> None:
        probabilities = [1.0 - index / 100.0 for index in range(100)]
        labels = [1] * 20 + [0] * 80
        audit = audit_risk_controlled_grid(
            probabilities,
            labels,
            coverages=(0.1, 0.2, 0.5),
            alpha=0.2,
            delta=0.05,
        )
        self.assertEqual([row["selected_count"] for row in audit], [10, 20, 50])
        self.assertEqual([row["errors"] for row in audit], [0, 0, 30])
        self.assertTrue(all(row["per_candidate_delta"] == 0.05 / 3 for row in audit))
        self.assertFalse(audit[-1]["certified"])

    def test_empirical_bernstein_tightens_with_more_zero_losses(self) -> None:
        small = empirical_bernstein_upper_confidence([0.0] * 20, 0.01)
        large = empirical_bernstein_upper_confidence([0.0] * 200, 0.01)
        self.assertLess(large, small)

    def test_empirical_bernstein_lower_bound_tightens_with_sample_size(self) -> None:
        small = empirical_bernstein_lower_confidence(
            [0.4] * 40,
            0.01,
            lower_bound=-1.4,
            upper_bound=1.0,
        )
        large = empirical_bernstein_lower_confidence(
            [0.4] * 4000,
            0.01,
            lower_bound=-1.4,
            upper_bound=1.0,
        )
        self.assertGreater(large, small)
        self.assertGreater(large, 0.0)

    def test_empirical_bernstein_lower_bound_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            empirical_bernstein_lower_confidence(
                [1.1],
                0.05,
                lower_bound=-1.4,
                upper_bound=1.0,
            )

    def test_bounded_grid_selects_low_shortfall_prefix(self) -> None:
        scores = [1.0 - index / 1000.0 for index in range(1000)]
        losses = [0.0] * 200 + [1.0] * 800
        selection = select_bounded_risk_grid(
            scores,
            losses,
            coverages=(0.1, 0.2, 0.5),
            alpha=0.1,
            delta=0.05,
        )
        self.assertEqual(selection.selected_count, 200)

    def test_hoeffding_tightens_with_more_zero_losses(self) -> None:
        small = hoeffding_upper_confidence([0.0] * 20, 0.01)
        large = hoeffding_upper_confidence([0.0] * 200, 0.01)
        self.assertLess(large, small)

    def test_hoeffding_grid_certifies_zero_loss_prefix(self) -> None:
        scores = [1.0 - index / 1000.0 for index in range(200)]
        losses = [0.0] * 100 + [1.0] * 100
        selection = select_hoeffding_risk_grid(
            scores,
            losses,
            coverages=(0.25, 0.5, 1.0),
            alpha=0.20,
            delta=0.05,
        )
        self.assertGreater(selection.selected_count, 0)
        self.assertLessEqual(selection.risk_upper_bound, 0.20)


if __name__ == "__main__":
    unittest.main()
