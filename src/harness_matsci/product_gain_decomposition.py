from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProductGainVariance:
    mean_success_probability: float
    mean_value_magnitude: float
    mean_net_gain: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    total_variance: float
    internal_samples: int
    external_samples: int
    cost: float

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)


def decompose_product_gain(
    success_probabilities: Sequence[float],
    value_magnitudes: Sequence[float],
    *,
    cost: float = 0.0,
    value_weights: Sequence[float] | None = None,
) -> ProductGainVariance:
    """Exactly decompose uncertainty in comparative scientific net gain.

    For a pairwise action, let ``X = 2 P(success) - 1`` be the signed success
    factor and ``M >= 0`` the scientific value gap between candidates. Under the
    crossed empirical product distribution ``G = X * M - cost``, the population
    variance decomposes exactly into

    ``E[M]^2 Var(X) + E[X]^2 Var(M) + Var(X) Var(M)``.

    The terms are interpreted as internal decision uncertainty, external value
    uncertainty, and their interaction. No Gaussian or independence assumption
    is needed for the empirical crossed grid because every internal sample is
    explicitly crossed with every external sample.
    """
    probabilities = tuple(float(value) for value in success_probabilities)
    magnitudes = tuple(float(value) for value in value_magnitudes)
    if len(probabilities) < 2 or len(magnitudes) < 2:
        raise ValueError("at least two internal and two external samples are required")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("success probabilities must be finite and lie in [0, 1]")
    if any(not math.isfinite(value) or value < 0.0 for value in magnitudes):
        raise ValueError("value magnitudes must be finite and nonnegative")
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("cost must be finite and nonnegative")
    weights = _normalized_weights(value_weights, len(magnitudes))
    signed_success = tuple(2.0 * value - 1.0 for value in probabilities)
    mean_success = _mean(signed_success)
    mean_magnitude = _weighted_mean(magnitudes, weights)
    success_variance = _variance(signed_success)
    magnitude_variance = _weighted_variance(magnitudes, weights)
    internal = mean_magnitude**2 * success_variance
    external = mean_success**2 * magnitude_variance
    interaction = success_variance * magnitude_variance
    total = _weighted_variance(
        tuple(
            success * magnitude - cost
            for success in signed_success
            for magnitude in magnitudes
        ),
        tuple(weight / len(signed_success) for _ in signed_success for weight in weights),
    )
    if not math.isclose(total, internal + external + interaction, abs_tol=1e-12):
        raise AssertionError("product gain variance decomposition is not exact")
    return ProductGainVariance(
        mean_success_probability=_mean(probabilities),
        mean_value_magnitude=mean_magnitude,
        mean_net_gain=mean_success * mean_magnitude - cost,
        internal_variance=internal,
        external_variance=external,
        interaction_variance=interaction,
        total_variance=total,
        internal_samples=len(probabilities),
        external_samples=len(magnitudes),
        cost=cost,
    )


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _variance(values) -> float:
    rows = list(values)
    center = _mean(rows)
    return _mean((value - center) ** 2 for value in rows)


def _normalized_weights(values: Sequence[float] | None, count: int) -> tuple[float, ...]:
    if values is None:
        return tuple(1.0 / count for _ in range(count))
    weights = tuple(float(value) for value in values)
    if len(weights) != count:
        raise ValueError("value weights must align with magnitudes")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("value weights must be finite and nonnegative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("value weights must have positive mass")
    return tuple(value / total for value in weights)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights))


def _weighted_variance(values: Sequence[float], weights: Sequence[float]) -> float:
    center = _weighted_mean(values, weights)
    return sum(weight * (value - center) ** 2 for value, weight in zip(values, weights))
