from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConformalGainObservation:
    record_id: str
    predicted_mean_gain: float
    predicted_scale: float
    observed_gain: float


@dataclass(frozen=True)
class ConformalGainBound:
    record_id: str
    predicted_mean_gain: float
    predicted_scale: float
    lower_gain_bound: float
    conformal_radius: float
    alpha: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class SplitConformalGainCalibrator:
    alpha: float
    normalized_radius: float
    calibration_count: int
    minimum_scale: float

    def predict_one(
        self,
        *,
        record_id: str,
        predicted_mean_gain: float,
        predicted_scale: float,
    ) -> ConformalGainBound:
        scale = max(self.minimum_scale, float(predicted_scale))
        radius = self.normalized_radius * scale
        return ConformalGainBound(
            record_id=record_id,
            predicted_mean_gain=float(predicted_mean_gain),
            predicted_scale=scale,
            lower_gain_bound=float(predicted_mean_gain) - radius,
            conformal_radius=radius,
            alpha=self.alpha,
        )

    def predict(
        self,
        rows: Iterable[tuple[str, float, float]],
    ) -> list[ConformalGainBound]:
        return [
            self.predict_one(
                record_id=record_id,
                predicted_mean_gain=predicted_mean_gain,
                predicted_scale=predicted_scale,
            )
            for record_id, predicted_mean_gain, predicted_scale in rows
        ]

    def to_json(self) -> dict[str, float | int | str]:
        return {
            "type": "one_sided_normalized_split_conformal_gain",
            "alpha": self.alpha,
            "normalized_radius": self.normalized_radius,
            "calibration_count": self.calibration_count,
            "minimum_scale": self.minimum_scale,
            "guarantee": "marginal_lower_gain_coverage_under_exchangeability",
        }


def fit_split_conformal_gain(
    observations: list[ConformalGainObservation],
    *,
    alpha: float = 0.1,
    minimum_scale: float = 1e-6,
) -> SplitConformalGainCalibrator:
    """Fit a finite-sample one-sided lower prediction bound for net gain.

    The nonconformity score is the scale-normalized overprediction error
    ``(predicted_mean - observed_gain) / predicted_scale``. The conformal
    finite-sample correction uses the ceil((n + 1) * (1 - alpha))-th order
    statistic. The resulting lower bound has marginal coverage at least
    ``1-alpha`` under exchangeability, without assuming Gaussian residuals.
    """
    if not observations:
        raise ValueError("split conformal calibration requires observations")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be finite and positive")
    scores = []
    for observation in observations:
        values = (
            observation.predicted_mean_gain,
            observation.predicted_scale,
            observation.observed_gain,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("conformal observations must be finite")
        scale = max(minimum_scale, float(observation.predicted_scale))
        scores.append(
            (float(observation.predicted_mean_gain) - float(observation.observed_gain))
            / scale
        )
    scores.sort()
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    rank = min(len(scores), max(1, rank))
    return SplitConformalGainCalibrator(
        alpha=alpha,
        normalized_radius=float(scores[rank - 1]),
        calibration_count=len(scores),
        minimum_scale=minimum_scale,
    )
