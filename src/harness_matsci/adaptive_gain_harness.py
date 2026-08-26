from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .source_decomposition import decompose_source_variance


@dataclass(frozen=True)
class CrossedNumericEstimate:
    mean: float
    positive_probability: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    epistemic_variance: float
    aleatoric_variance: float
    predictive_variance: float
    model_replicas: int
    evidence_replicas: int

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class NumericActionWorthiness:
    record_id: str
    benchmark: str
    stage: str
    expected_net_gain: float
    p_positive_gain: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    aleatoric_variance: float
    predictive_variance: float
    conservative_gain: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class NumericEvidenceValue:
    record_id: str
    benchmark: str
    evidence_stage: str
    expected_value: float
    p_positive_value: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    aleatoric_variance: float
    predictive_variance: float
    conservative_value: float
    evidence_cost: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


def crossed_numeric_estimate(
    cells: Sequence[Sequence[float]],
    *,
    aleatoric_variance: float = 0.0,
) -> CrossedNumericEstimate:
    matrix = _validated_matrix(cells)
    decomposition = decompose_source_variance(matrix)
    flattened = [value for row in matrix for value in row]
    epistemic_variance = decomposition.total_variance
    irreducible = max(0.0, float(aleatoric_variance))
    return CrossedNumericEstimate(
        mean=sum(flattened) / len(flattened),
        positive_probability=sum(value > 0.0 for value in flattened) / len(flattened),
        internal_variance=decomposition.internal_variance,
        external_variance=decomposition.external_variance,
        interaction_variance=decomposition.interaction_variance,
        epistemic_variance=epistemic_variance,
        aleatoric_variance=irreducible,
        predictive_variance=epistemic_variance + irreducible,
        model_replicas=decomposition.agent_replicas,
        evidence_replicas=decomposition.evidence_views,
    )


def counterfactual_evidence_value_cells(
    current_gain_by_model: Sequence[float],
    future_gain_cells: Sequence[Sequence[float]],
    *,
    evidence_cost: float,
    cost_weight: float,
    current_abstention_value: float = 0.0,
    future_abstention_value: float = 0.0,
) -> list[list[float]]:
    matrix = _validated_matrix(future_gain_cells)
    current = [float(value) for value in current_gain_by_model]
    if len(current) != len(matrix):
        raise ValueError("current and future predictions need the same model replicas")
    charged_cost = max(0.0, float(cost_weight)) * max(0.0, float(evidence_cost))
    return [
        [
            max(float(future_abstention_value), future_gain)
            - max(float(current_abstention_value), current_gain)
            - charged_cost
            for future_gain in future_row
        ]
        for current_gain, future_row in zip(current, matrix)
    ]


def sunk_cost_neutral_score_cells(
    gain_cells: Sequence[Sequence[float]],
    *,
    cumulative_evidence_cost: float,
    cost_weight: float,
    lower_bound_radius: float = 0.0,
) -> list[list[float]]:
    matrix = _validated_matrix(gain_cells)
    sunk_cost = max(0.0, float(cost_weight)) * max(
        0.0, float(cumulative_evidence_cost)
    )
    radius = max(0.0, float(lower_bound_radius))
    return [
        [value + sunk_cost - radius for value in row]
        for row in matrix
    ]


def leave_one_out_topk_shadow_prices(
    scores: Sequence[float],
    *,
    budget: int,
    floor: float = 0.0,
) -> list[float]:
    values = [float(value) for value in scores]
    if not values:
        raise ValueError("shadow prices require at least one score")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("shadow-price scores must be finite")
    if budget < 1:
        raise ValueError("top-k budget must be positive")
    effective_budget = min(int(budget), len(values))
    ordered = sorted(
        range(len(values)),
        key=lambda index: (-values[index], index),
    )
    rank = {index: position for position, index in enumerate(ordered)}
    kth_score = values[ordered[effective_budget - 1]]
    replacement_score = (
        values[ordered[effective_budget]]
        if effective_budget < len(values)
        else float(floor)
    )
    lower_floor = float(floor)
    return [
        max(lower_floor, replacement_score if rank[index] < effective_budget else kth_score)
        for index in range(len(values))
    ]


def conservative_score(mean: float, variance: float, radius: float) -> float:
    return float(mean) - max(0.0, float(radius)) * math.sqrt(
        max(0.0, float(variance))
    )


def _validated_matrix(cells: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = [[float(value) for value in row] for row in cells]
    if len(matrix) < 2:
        raise ValueError("at least two model replicas are required")
    widths = {len(row) for row in matrix}
    if len(widths) != 1 or not widths or next(iter(widths)) < 2:
        raise ValueError("a balanced matrix with at least two evidence replicas is required")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("crossed predictions must be finite")
    return matrix
