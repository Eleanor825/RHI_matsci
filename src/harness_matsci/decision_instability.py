from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecisionBoundaryInstability:
    positive_fraction: float
    normalized_vote_entropy: float
    value_range: float
    boundary_instability: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


def decision_boundary_instability(
    values: list[float],
    *,
    weights: list[float] | None = None,
) -> DecisionBoundaryInstability:
    """Measure instability specifically at the execute/abstain boundary.

    Variance treats two equally dispersed ensembles the same even when only one
    crosses zero net scientific gain. This statistic multiplies normalized vote
    entropy by squared value range, so it is zero for sign-consistent replicas
    and grows when scientifically meaningful decisions flip.
    """
    if len(values) < 2:
        raise ValueError("decision instability requires at least two values")
    if weights is None:
        normalized_weights = [1.0 / len(values)] * len(values)
    else:
        if len(weights) != len(values):
            raise ValueError("weights must align with values")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("weights must be nonnegative")
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("weights must have positive mass")
        normalized_weights = [weight / total for weight in weights]
    positive_fraction = sum(
        weight for value, weight in zip(values, normalized_weights) if value > 0.0
    )
    entropy = _binary_entropy(positive_fraction)
    value_range = max(values) - min(values)
    return DecisionBoundaryInstability(
        positive_fraction=positive_fraction,
        normalized_vote_entropy=entropy,
        value_range=value_range,
        boundary_instability=entropy * value_range * value_range,
    )


def _binary_entropy(probability: float) -> float:
    probability = min(1.0, max(0.0, probability))
    if probability in (0.0, 1.0):
        return 0.0
    entropy = -probability * math.log(probability) - (1.0 - probability) * math.log(
        1.0 - probability
    )
    return entropy / math.log(2.0)
