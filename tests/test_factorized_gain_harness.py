import pytest

from harness_matsci.factorized_gain_harness import (
    FactorizedGainState,
    crossed_grid_decomposition,
    fit_factorized_gain_harness,
    fit_factorized_temperature,
)


def _state(index, *, success, utility, gain, low_utility=False):
    base_utility = 0.42 if low_utility else 0.85
    source_utility = 0.4 if low_utility else 0.88
    return FactorizedGainState(
        record_id=f"r{index}",
        benchmark="task",
        objective="task",
        group_id=f"g{index % 6}",
        model_worthiness_probabilities=(0.25, 0.3, 0.35),
        model_success_probabilities=(0.75, 0.8, 0.85),
        model_utility_means=(base_utility - 0.03, base_utility, base_utility + 0.03),
        source_names=("screening", "verification"),
        source_success_probabilities=(0.78, 0.86),
        source_utility_features=(
            (source_utility - 0.05, source_utility - 0.1, 0.2),
            (source_utility, source_utility - 0.03, 0.3),
        ),
        source_weights=(0.55, 0.9),
        normalized_cost=0.2,
        failure_harm=0.2,
        reservation_value=0.7,
        actual_success=success,
        actual_normalized_utility=utility,
        actual_net_gain=gain,
    )


def test_crossed_grid_decomposition_is_exact():
    decomposition = crossed_grid_decomposition(
        ((0.2, 0.4), (0.3, 0.8), (0.5, 0.9)),
        (0.4, 0.6),
    )
    assert decomposition["total"] == pytest.approx(
        decomposition["internal"]
        + decomposition["external"]
        + decomposition["interaction"]
    )


def test_factorization_separates_success_from_scientific_value():
    states = []
    for index in range(36):
        high = index % 3 != 0
        states.append(
            _state(
                index,
                success=1,
                utility=0.9 if high else 0.4,
                gain=0.17 if high else -0.33,
                low_utility=not high,
            )
        )
    model = fit_factorized_gain_harness(
        states,
        blend_weight=1.0,
        regularization=0.1,
    )
    model = fit_factorized_temperature(model, states)
    high = model.predict_one(_state(100, success=1, utility=0.9, gain=0.17))
    low = model.predict_one(
        _state(101, success=1, utility=0.4, gain=-0.33, low_utility=True)
    )
    assert high.success_probability > 0.5
    assert low.success_probability > 0.5
    assert high.action_worthiness > low.action_worthiness
    assert high.expected_net_gain > low.expected_net_gain
    assert high.total_gain_variance == pytest.approx(
        high.aleatoric_gain_variance
        + high.gain_internal_variance
        + high.gain_external_variance
        + high.gain_interaction_variance
    )
