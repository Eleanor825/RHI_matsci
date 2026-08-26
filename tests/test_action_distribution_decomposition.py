import math

import pytest

from harness_matsci.action_distribution_decomposition import (
    categorical_action_grid,
    decompose_action_distribution,
)


def test_exactly_decomposes_internal_external_and_interaction_variance():
    grid = (
        ((0.9, 0.1), (0.7, 0.3), (0.6, 0.4)),
        ((0.4, 0.6), (0.3, 0.7), (0.8, 0.2)),
        ((0.5, 0.5), (0.2, 0.8), (0.1, 0.9)),
    )

    result = decompose_action_distribution(grid, action_names=("A", "B"))

    assert math.isclose(
        result.total_variance,
        result.internal_variance
        + result.external_variance
        + result.interaction_variance,
        abs_tol=1e-12,
    )
    assert result.internal_variance > 0.0
    assert result.external_variance > 0.0
    assert result.interaction_variance > 0.0
    assert result.mean_distribution == pytest.approx((0.5, 0.5))
    assert result.consensus_uncertainty == pytest.approx(0.5)


def test_identifies_pure_internal_variation():
    grid = categorical_action_grid(
        (("A", "A", "A"), ("B", "B", "B")),
        action_names=("A", "B"),
    )

    result = decompose_action_distribution(grid, action_names=("A", "B"))

    assert result.internal_variance == pytest.approx(result.total_variance)
    assert result.external_variance == pytest.approx(0.0)
    assert result.interaction_variance == pytest.approx(0.0)


def test_identifies_pure_external_variation():
    grid = categorical_action_grid(
        (("A", "B"), ("A", "B"), ("A", "B")),
        action_names=("A", "B"),
    )

    result = decompose_action_distribution(grid, action_names=("A", "B"))

    assert result.external_variance == pytest.approx(result.total_variance)
    assert result.internal_variance == pytest.approx(0.0)
    assert result.interaction_variance == pytest.approx(0.0)


def test_identifies_replica_evidence_interaction():
    grid = categorical_action_grid(
        (("A", "B"), ("B", "A")),
        action_names=("A", "B"),
    )

    result = decompose_action_distribution(grid, action_names=("A", "B"))

    assert result.interaction_variance == pytest.approx(result.total_variance)
    assert result.internal_variance == pytest.approx(0.0)
    assert result.external_variance == pytest.approx(0.0)


def test_rejects_unbalanced_or_invalid_probability_grids():
    with pytest.raises(ValueError, match="balanced grid"):
        decompose_action_distribution((((0.5, 0.5),), ((0.5, 0.5),)))
    with pytest.raises(ValueError, match="sum to one"):
        decompose_action_distribution(
            (((0.6, 0.6), (0.5, 0.5)), ((0.5, 0.5), (0.5, 0.5)))
        )
