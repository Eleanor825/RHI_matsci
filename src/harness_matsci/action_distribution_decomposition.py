from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ActionDistributionVariance:
    action_names: tuple[str, ...]
    mean_distribution: tuple[float, ...]
    internal_variance: float
    external_variance: float
    interaction_variance: float
    total_variance: float
    consensus_uncertainty: float
    internal_replicas: int
    external_views: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def decompose_action_distribution(
    distributions: Sequence[Sequence[Sequence[float]]],
    *,
    action_names: Sequence[str] | None = None,
) -> ActionDistributionVariance:
    """Decompose a balanced replica-by-evidence grid of action distributions.

    Each cell is a categorical probability vector over the same action menu.
    The squared Euclidean population variance of the full grid is decomposed
    exactly into internal-replica, external-evidence, and interaction effects.
    One-hot vectors may be used when only sampled action choices are available.
    """
    grid = _validated_grid(distributions)
    replica_count = len(grid)
    evidence_count = len(grid[0])
    action_count = len(grid[0][0])
    names = tuple(action_names) if action_names is not None else tuple(
        f"action_{index}" for index in range(action_count)
    )
    if len(names) != action_count or len(set(names)) != action_count:
        raise ValueError("action_names must be unique and align with distribution width")

    replica_means = [
        _vector_mean(grid[replica_index])
        for replica_index in range(replica_count)
    ]
    evidence_means = [
        _vector_mean(
            [grid[replica_index][evidence_index] for replica_index in range(replica_count)]
        )
        for evidence_index in range(evidence_count)
    ]
    grand_mean = _vector_mean(replica_means)

    internal = _mean(
        _squared_distance(replica_mean, grand_mean)
        for replica_mean in replica_means
    )
    external = _mean(
        _squared_distance(evidence_mean, grand_mean)
        for evidence_mean in evidence_means
    )
    interaction = _mean(
        _squared_norm(
            tuple(
                grid[replica_index][evidence_index][action_index]
                - replica_means[replica_index][action_index]
                - evidence_means[evidence_index][action_index]
                + grand_mean[action_index]
                for action_index in range(action_count)
            )
        )
        for replica_index in range(replica_count)
        for evidence_index in range(evidence_count)
    )
    total = _mean(
        _squared_distance(cell, grand_mean)
        for row in grid
        for cell in row
    )
    if not math.isclose(total, internal + external + interaction, abs_tol=1e-12):
        raise AssertionError("action-distribution variance decomposition is not exact")

    return ActionDistributionVariance(
        action_names=names,
        mean_distribution=grand_mean,
        internal_variance=internal,
        external_variance=external,
        interaction_variance=interaction,
        total_variance=total,
        consensus_uncertainty=1.0 - max(grand_mean),
        internal_replicas=replica_count,
        external_views=evidence_count,
    )


def categorical_action_grid(
    choices: Sequence[Sequence[str]],
    *,
    action_names: Sequence[str] | None = None,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    if len(choices) < 2:
        raise ValueError("at least two internal replicas are required")
    widths = {len(row) for row in choices}
    if len(widths) != 1 or not widths or next(iter(widths)) < 2:
        raise ValueError("a balanced grid with at least two external views is required")
    names = tuple(action_names) if action_names is not None else tuple(
        sorted({choice for row in choices for choice in row})
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("action_names must be non-empty and unique")
    index = {name: position for position, name in enumerate(names)}
    output = []
    for row in choices:
        encoded_row = []
        for choice in row:
            if choice not in index:
                raise ValueError(f"unknown action choice: {choice}")
            encoded_row.append(
                tuple(
                    1.0 if action_index == index[choice] else 0.0
                    for action_index in range(len(names))
                )
            )
        output.append(tuple(encoded_row))
    return tuple(output)


def _validated_grid(
    distributions: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    if len(distributions) < 2:
        raise ValueError("at least two internal replicas are required")
    evidence_widths = {len(row) for row in distributions}
    if len(evidence_widths) != 1 or not evidence_widths or next(iter(evidence_widths)) < 2:
        raise ValueError("a balanced grid with at least two external views is required")
    action_widths = {len(cell) for row in distributions for cell in row}
    if len(action_widths) != 1 or not action_widths or next(iter(action_widths)) < 2:
        raise ValueError("each cell must contain the same action distribution")
    output = []
    for row in distributions:
        normalized_row = []
        for cell in row:
            values = tuple(float(value) for value in cell)
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("action probabilities must be finite and nonnegative")
            total = sum(values)
            if not math.isclose(total, 1.0, abs_tol=1e-8):
                raise ValueError("each action distribution must sum to one")
            normalized_row.append(tuple(value / total for value in values))
        output.append(tuple(normalized_row))
    return tuple(output)


def _vector_mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(
        _mean(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    )


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right)
    )


def _squared_norm(values: Sequence[float]) -> float:
    return sum(value**2 for value in values)


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0
