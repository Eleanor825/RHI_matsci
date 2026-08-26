import pytest

from harness_matsci.product_gain_decomposition import decompose_product_gain


def test_product_decomposition_matches_explicit_crossed_grid():
    result = decompose_product_gain(
        (0.6, 0.8, 0.9),
        (0.1, 0.4, 0.7, 0.9),
        cost=0.02,
    )

    assert result.total_variance == pytest.approx(
        result.internal_variance
        + result.external_variance
        + result.interaction_variance
    )
    assert result.mean_net_gain == pytest.approx(
        (2 * result.mean_success_probability - 1)
        * result.mean_value_magnitude
        - 0.02
    )
    assert result.internal_variance > 0.0
    assert result.external_variance > 0.0
    assert result.interaction_variance > 0.0


def test_internal_and_external_terms_have_intervention_semantics():
    internal_only = decompose_product_gain((0.55, 0.95), (0.4, 0.4))
    external_only = decompose_product_gain((0.8, 0.8), (0.1, 0.7))

    assert internal_only.internal_variance > 0.0
    assert internal_only.external_variance == pytest.approx(0.0)
    assert internal_only.interaction_variance == pytest.approx(0.0)
    assert external_only.external_variance > 0.0
    assert external_only.internal_variance == pytest.approx(0.0)
    assert external_only.interaction_variance == pytest.approx(0.0)


def test_supports_reliability_weighted_external_sources():
    uniform = decompose_product_gain((0.7, 0.9), (0.1, 0.9))
    weighted = decompose_product_gain(
        (0.7, 0.9),
        (0.1, 0.9),
        value_weights=(0.9, 0.1),
    )

    assert weighted.mean_value_magnitude < uniform.mean_value_magnitude
    assert weighted.total_variance == pytest.approx(
        weighted.internal_variance
        + weighted.external_variance
        + weighted.interaction_variance
    )


def test_rejects_invalid_samples():
    with pytest.raises(ValueError, match="at least two"):
        decompose_product_gain((0.5,), (0.2, 0.3))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decompose_product_gain((0.5, 1.2), (0.2, 0.3))
