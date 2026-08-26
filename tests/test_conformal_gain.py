import pytest

from harness_matsci.conformal_gain import (
    ConformalGainObservation,
    fit_split_conformal_gain,
)


def test_uses_finite_sample_corrected_one_sided_quantile():
    observations = [
        ConformalGainObservation(
            record_id=str(index),
            predicted_mean_gain=float(index),
            predicted_scale=1.0,
            observed_gain=0.0,
        )
        for index in range(9)
    ]

    calibrator = fit_split_conformal_gain(observations, alpha=0.2)

    assert calibrator.normalized_radius == pytest.approx(7.0)
    bound = calibrator.predict_one(
        record_id="held-out",
        predicted_mean_gain=10.0,
        predicted_scale=2.0,
    )
    assert bound.conformal_radius == pytest.approx(14.0)
    assert bound.lower_gain_bound == pytest.approx(-4.0)


def test_normalizes_overprediction_by_predicted_scale():
    observations = [
        ConformalGainObservation("a", 0.8, 0.2, 0.4),
        ConformalGainObservation("b", 0.8, 0.4, 0.4),
        ConformalGainObservation("c", 0.8, 0.8, 0.4),
    ]

    calibrator = fit_split_conformal_gain(observations, alpha=0.5)

    assert calibrator.normalized_radius == pytest.approx(1.0)


def test_allows_negative_radius_when_model_is_uniformly_conservative():
    observations = [
        ConformalGainObservation("a", 0.1, 1.0, 0.4),
        ConformalGainObservation("b", 0.2, 1.0, 0.6),
        ConformalGainObservation("c", 0.3, 1.0, 0.8),
    ]

    calibrator = fit_split_conformal_gain(observations, alpha=0.5)
    bound = calibrator.predict_one(
        record_id="held-out",
        predicted_mean_gain=0.2,
        predicted_scale=1.0,
    )

    assert calibrator.normalized_radius < 0.0
    assert bound.lower_gain_bound > bound.predicted_mean_gain


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="requires observations"):
        fit_split_conformal_gain([])
    with pytest.raises(ValueError, match="alpha"):
        fit_split_conformal_gain(
            [ConformalGainObservation("a", 0.0, 1.0, 0.0)], alpha=1.0
        )
