from __future__ import annotations

import unittest

from harness_matsci.schema import ActionRecord
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    EstimatedGainProjector,
    GainConfig,
    ScientificShortfallScales,
    gain_targets,
)


def _record(record_id: str, utility: float, label: int, *, cost: float, reversibility: float) -> ActionRecord:
    return ActionRecord(
        record_id=record_id,
        benchmark="task",
        split="train",
        visible_context="context",
        candidate_action="act",
        action_type="choose_candidate",
        evidence=[],
        features={"cost": cost, "reversibility": reversibility},
        label=label,
        utility=utility,
    )


class ScientificGainTests(unittest.TestCase):
    def test_continuous_realized_utility_preserves_near_miss_value(self):
        record = _record(
            "near-miss", utility=0.6, label=0, cost=0.5, reversibility=0.5
        )
        records = [
            _record("low", utility=0.0, label=0, cost=0.5, reversibility=0.5),
            record,
            _record("high", utility=1.0, label=1, cost=0.5, reversibility=0.5),
        ]
        normalizer = EmpiricalUtilityNormalizer.fit(records)
        gated = gain_targets(records, normalizer, GainConfig())[1]
        continuous = gain_targets(
            records,
            normalizer,
            GainConfig(utility_mode="continuous_realized"),
        )[1]
        self.assertGreater(continuous.base_gain, gated.base_gain)

    def test_unknown_utility_mode_is_rejected(self):
        record = _record(
            "record", utility=0.5, label=1, cost=0.5, reversibility=0.5
        )
        with self.assertRaises(ValueError):
            gain_targets(
                [record],
                EmpiricalUtilityNormalizer.fit([record]),
                GainConfig(utility_mode="unknown"),
            )

    def test_estimated_gain_projector_uses_train_only_contract(self):
        train = [
            _record("low", 0.1, 0, cost=0.5, reversibility=0.5),
            _record("mid", 0.5, 1, cost=0.5, reversibility=0.5),
            _record("high", 0.9, 1, cost=0.5, reversibility=0.5),
        ]
        normalizer = EmpiricalUtilityNormalizer.fit(train)
        config = GainConfig(utility_mode="continuous_realized")
        reservations = BudgetReservationValues.fit(
            train,
            normalizer,
            config,
            budget_fraction=0.34,
            strategy="potential_success_slot_value",
        )
        projector = EstimatedGainProjector(normalizer, reservations, config)
        record = _record("candidate", 0.0, 0, cost=0.5, reversibility=0.5)
        low = projector.project(
            record,
            estimated_utility=0.1,
            success_probability=0.1,
        )
        high = projector.project(
            record,
            estimated_utility=0.9,
            success_probability=0.9,
        )
        self.assertGreater(high, low)

        normalized_utility = normalizer.encode_value(record.benchmark, 0.9)
        cost = record.features["cost"]
        irreversibility = 1.0 - record.features["reversibility"]
        failure_harm = 0.5 * cost + 0.5 * irreversibility
        expected = (
            0.9 * normalized_utility
            - config.cost_weight * cost
            - config.failure_harm_weight * 0.1 * failure_harm
            - reservations.value(record.benchmark)
        )
        self.assertAlmostEqual(high, expected)

    def test_gain_uses_train_fitted_empirical_utility(self) -> None:
        train = [
            _record("a", 0.0, 0, cost=0.2, reversibility=0.8),
            _record("b", 1.0, 1, cost=0.2, reversibility=0.8),
            _record("c", 2.0, 1, cost=0.2, reversibility=0.8),
        ]
        normalizer = EmpiricalUtilityNormalizer.fit(train)
        targets = gain_targets(train, normalizer, GainConfig())
        self.assertLess(targets[0].gain, 0.0)
        self.assertGreater(targets[1].gain, 0.0)
        self.assertGreater(targets[2].normalized_utility, targets[1].normalized_utility)

    def test_failed_irreversible_action_has_larger_negative_gain(self) -> None:
        records = [
            _record("safe", 0.0, 0, cost=0.2, reversibility=0.9),
            _record("risky", 0.0, 0, cost=0.8, reversibility=0.1),
        ]
        targets = gain_targets(records, EmpiricalUtilityNormalizer.fit(records), GainConfig())
        self.assertLess(targets[1].gain, targets[0].gain)

    def test_budget_reservation_creates_opportunity_adjusted_worthiness(self) -> None:
        records = [
            _record(str(index), float(index), 1, cost=0.2, reversibility=0.8)
            for index in range(10)
        ]
        normalizer = EmpiricalUtilityNormalizer.fit(records)
        config = GainConfig()
        reservation = BudgetReservationValues.fit(
            records, normalizer, config, budget_fraction=0.2
        )
        targets = gain_targets(records, normalizer, config, reservation)
        self.assertEqual(sum(target.worthy for target in targets), 2)

    def test_shortfall_preserves_failure_magnitude(self) -> None:
        records = [
            _record(str(index), float(index), int(index >= 5), cost=0.2, reversibility=0.8)
            for index in range(10)
        ]
        normalizer = EmpiricalUtilityNormalizer.fit(records)
        config = GainConfig()
        reservation = BudgetReservationValues.fit(
            records, normalizer, config, budget_fraction=0.2
        )
        targets = gain_targets(records, normalizer, config, reservation)
        scales = ScientificShortfallScales.fit(targets)
        ordered = sorted(targets, key=lambda target: target.gain)
        self.assertGreater(scales.loss(ordered[0]), scales.loss(ordered[-2]))
        self.assertEqual(scales.loss(ordered[-1]), 0.0)

    def test_reservation_respects_zero_gain_outside_option(self) -> None:
        records = [
            _record(str(index), float(index), int(index == 9), cost=0.2, reversibility=0.8)
            for index in range(10)
        ]
        normalizer = EmpiricalUtilityNormalizer.fit(records)
        reservation = BudgetReservationValues.fit(
            records, normalizer, GainConfig(), budget_fraction=0.2
        )
        self.assertGreaterEqual(reservation.value("task"), 0.0)
        self.assertEqual(reservation.outside_option, 0.0)

    def test_potential_success_reservation_does_not_jump_with_label_prevalence(self) -> None:
        mostly_positive = [
            _record(str(index), float(index), int(index >= 8), cost=0.2, reversibility=0.8)
            for index in range(10)
        ]
        fewer_positive = [
            _record(str(index), float(index), int(index == 9), cost=0.2, reversibility=0.8)
            for index in range(10)
        ]
        first_normalizer = EmpiricalUtilityNormalizer.fit(mostly_positive)
        second_normalizer = EmpiricalUtilityNormalizer.fit(fewer_positive)
        first = BudgetReservationValues.fit(
            mostly_positive,
            first_normalizer,
            GainConfig(),
            budget_fraction=0.2,
            strategy="potential_success_slot_value",
        )
        second = BudgetReservationValues.fit(
            fewer_positive,
            second_normalizer,
            GainConfig(),
            budget_fraction=0.2,
            strategy="potential_success_slot_value",
        )
        self.assertAlmostEqual(first.value("task"), second.value("task"))
        self.assertEqual(first.strategy, "potential_success_slot_value")


if __name__ == "__main__":
    unittest.main()
