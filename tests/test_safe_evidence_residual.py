import math

import pytest

from harness_matsci.safe_evidence_residual import (
    aggregate_probability_grid_decomposition,
    EvidenceResidualParameters,
    EvidenceResidualState,
    SafeEvidenceResidualModel,
    TemperatureCalibratedEvidenceModel,
    fit_evidence_temperature,
    fit_safe_aggregate_evidence_residual,
    fit_safe_evidence_residual,
    probability_grid_decomposition,
)
from harness_matsci.schema import ActionRecord


def _state(index: int, gain: float, source_shift: float = 0.0):
    probability = 0.75 if gain > 0.0 else 0.55
    return EvidenceResidualState(
        record_id=f"r{index}",
        benchmark="task",
        objective="objective",
        candidate_group_ids=(f"a{index}", f"b{index}"),
        model_success_probabilities=(probability - 0.05, probability, probability + 0.05),
        proposal_aligned_source_matrix=(
            (source_shift - 0.1, source_shift + 0.1),
            (source_shift - 0.05, source_shift + 0.15),
        ),
        source_names=("s1", "s2"),
        source_weights=(0.4, 0.6),
        actual_net_gain=gain,
    )


def test_weighted_probability_decomposition_is_exact():
    model = SafeEvidenceResidualModel(
        EvidenceResidualParameters(
            task_biases={"objective": 0.0},
            source_center=0.0,
            source_scale=1.0,
            source_coefficient=1.0,
            blend_weight=0.5,
            positive_intercept=0.2,
            positive_slope=0.1,
            negative_intercept=0.2,
            negative_slope=0.1,
            positive_residual_variance=0.01,
            negative_residual_variance=0.01,
            regularization=1.0,
        )
    )
    decomposition = probability_grid_decomposition(model, _state(0, 0.2, 0.2))

    assert decomposition["total"] == pytest.approx(
        decomposition["internal"]
        + decomposition["external"]
        + decomposition["interaction"]
    )


def test_zero_blend_preserves_direct_and_removes_external_variance():
    parameters = EvidenceResidualParameters(
        task_biases={"objective": 1.0},
        source_center=0.0,
        source_scale=1.0,
        source_coefficient=3.0,
        blend_weight=0.0,
        positive_intercept=0.2,
        positive_slope=0.0,
        negative_intercept=0.2,
        negative_slope=0.0,
        positive_residual_variance=0.01,
        negative_residual_variance=0.01,
        regularization=1.0,
    )
    model = SafeEvidenceResidualModel(parameters)
    state = _state(1, 0.2, 0.8)
    prediction = model.predict_one(state)
    decomposition = probability_grid_decomposition(model, state)

    assert prediction.action_worthiness == pytest.approx(
        sum(state.model_success_probabilities) / 3.0
    )
    assert decomposition["external"] == pytest.approx(0.0, abs=1e-15)
    assert decomposition["interaction"] == pytest.approx(0.0, abs=1e-15)


def test_positive_aligned_sources_raise_worthiness():
    parameters = EvidenceResidualParameters(
        task_biases={"objective": 0.0},
        source_center=0.0,
        source_scale=1.0,
        source_coefficient=2.0,
        blend_weight=1.0,
        positive_intercept=0.2,
        positive_slope=0.0,
        negative_intercept=0.2,
        negative_slope=0.0,
        positive_residual_variance=0.01,
        negative_residual_variance=0.01,
        regularization=1.0,
    )
    model = SafeEvidenceResidualModel(parameters)

    low = model.predict_one(_state(2, -0.2, -0.5))
    high = model.predict_one(_state(3, 0.2, 0.5))

    assert high.action_worthiness > low.action_worthiness


def test_fit_returns_finite_hurdle_distribution():
    states = [
        _state(index, 0.25 if index % 3 else -0.2, 0.4 if index % 3 else -0.4)
        for index in range(18)
    ]
    model = fit_safe_evidence_residual(
        states,
        blend_weight=0.5,
        regularization=1.0,
    )
    prediction = model.predict_one(states[0])

    assert 0.0 < prediction.action_worthiness < 1.0
    assert math.isfinite(prediction.expected_net_gain)
    assert prediction.total_variance >= prediction.irreducible_variance


def test_aggregate_residual_moment_matches_crossed_grid_mean():
    states = [
        _state(index, 0.25 if index % 3 else -0.2, 0.4 if index % 3 else -0.4)
        for index in range(18)
    ]
    model = fit_safe_aggregate_evidence_residual(
        states,
        blend_weight=0.5,
        regularization=1.0,
    )
    prediction = model.predict_one(states[1])
    decomposition = aggregate_probability_grid_decomposition(model, states[1])

    assert decomposition["mean"] == pytest.approx(prediction.action_worthiness)
    assert decomposition["total"] == pytest.approx(
        decomposition["internal"]
        + decomposition["external"]
        + decomposition["interaction"]
    )


def test_temperature_calibration_preserves_exact_variance_accounting():
    states = [
        _state(index, 0.25 if index % 3 else -0.2, 0.4 if index % 3 else -0.4)
        for index in range(18)
    ]
    base = fit_safe_aggregate_evidence_residual(
        states,
        blend_weight=0.5,
        regularization=1.0,
    )
    model = fit_evidence_temperature(base, states)
    prediction = model.predict_one(states[1])

    assert isinstance(model, TemperatureCalibratedEvidenceModel)
    assert prediction.total_variance == pytest.approx(
        prediction.irreducible_variance
        + prediction.internal_variance
        + prediction.external_variance
        + prediction.interaction_variance
    )
    assert 0.0 < prediction.action_worthiness < 1.0


def test_build_states_preserves_external_source_direction():
    from harness_matsci.safe_evidence_residual import build_evidence_residual_states

    record = ActionRecord(
        record_id="pair",
        benchmark="task",
        split="train",
        visible_context="A versus B",
        candidate_action="choose A",
        action_type="choose_candidate",
        evidence=[],
        features={},
        label=1,
        metadata={
            "objective": "objective",
            "candidate_group_ids": ["a", "b"],
            "hidden_outcome_for_evaluation_only": {"utility_difference": 0.4},
        },
    )
    state = build_evidence_residual_states(
        [record],
        [
            {
                "record_id": "pair",
                "proposal_sign": -1.0,
                "source_matrix": ((0.2, -0.3), (0.1, -0.4)),
            }
        ],
        {"pair": (0.7, 0.8)},
        source_names=("s1", "s2"),
        source_weights={"s1": 0.5, "s2": 0.5},
    )[0]

    assert state.proposal_aligned_source_matrix == ((-0.2, 0.3), (-0.1, 0.4))
    assert state.actual_net_gain == pytest.approx(0.39)
