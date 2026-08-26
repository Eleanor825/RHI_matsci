from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .risk_control import binomial_upper_confidence


@dataclass(frozen=True)
class FrozenPolicy:
    method: str
    threshold: float
    target_coverage: float
    selected_count: int
    empirical_risk: float
    population_gain_after_cost: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenPolicyCertificate:
    policy: FrozenPolicy
    selected_count: int
    coverage: float
    errors: int
    empirical_risk: float
    risk_upper_bound: float
    population_gain_after_cost: float
    selected_mean_gain_after_cost: float
    alpha: float
    delta: float
    certified: bool

    def to_json(self) -> dict[str, object]:
        output = asdict(self)
        output["policy"] = self.policy.to_json()
        return output


def select_feedback_policy(
    scores_by_method: dict[str, list[float]],
    labels: list[int],
    gains: list[float],
    tool_cost_by_method: dict[str, float],
    *,
    coverages: tuple[float, ...],
    max_empirical_risk: float = 0.20,
) -> FrozenPolicy:
    if not scores_by_method:
        raise ValueError("at least one policy method is required")
    if len(labels) != len(gains) or not labels:
        raise ValueError("labels and gains must be non-empty and aligned")
    if not coverages or any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("coverages must lie in (0, 1]")
    candidates = []
    for method, scores in sorted(scores_by_method.items()):
        if len(scores) != len(labels):
            raise ValueError(f"scores for {method} do not align with labels")
        if method not in tool_cost_by_method:
            raise ValueError(f"missing tool cost for {method}")
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        for coverage in coverages:
            count = max(1, min(len(order), math.ceil(len(order) * coverage)))
            selected = order[:count]
            errors = sum(1 - int(labels[index]) for index in selected)
            empirical_risk = errors / count
            if empirical_risk > max_empirical_risk:
                continue
            threshold = _prefix_threshold(scores, order, count)
            cost = tool_cost_by_method[method]
            population_gain = sum(
                gains[index] - cost for index in selected
            ) / len(scores)
            candidates.append(
                FrozenPolicy(
                    method=method,
                    threshold=threshold,
                    target_coverage=coverage,
                    selected_count=count,
                    empirical_risk=empirical_risk,
                    population_gain_after_cost=population_gain,
                )
            )
    positive = [candidate for candidate in candidates if candidate.population_gain_after_cost > 0.0]
    if not positive:
        return FrozenPolicy(
            method="no_op",
            threshold=math.inf,
            target_coverage=0.0,
            selected_count=0,
            empirical_risk=0.0,
            population_gain_after_cost=0.0,
        )
    return max(
        positive,
        key=lambda candidate: (
            candidate.population_gain_after_cost,
            candidate.selected_count,
            candidate.method,
        ),
    )


def certify_frozen_policy(
    policy: FrozenPolicy,
    scores: list[float],
    labels: list[int],
    gains: list[float],
    *,
    tool_cost: float,
    alpha: float,
    delta: float,
) -> FrozenPolicyCertificate:
    if len(scores) != len(labels) or len(scores) != len(gains) or not scores:
        raise ValueError("scores, labels, and gains must be non-empty and aligned")
    selected = [
        index for index, score in enumerate(scores) if score >= policy.threshold
    ]
    errors = sum(1 - int(labels[index]) for index in selected)
    upper = binomial_upper_confidence(errors, len(selected), delta)
    population_gain = sum(
        gains[index] - tool_cost for index in selected
    ) / len(scores)
    selected_mean = (
        sum(gains[index] - tool_cost for index in selected) / len(selected)
        if selected
        else 0.0
    )
    return FrozenPolicyCertificate(
        policy=policy,
        selected_count=len(selected),
        coverage=len(selected) / len(scores),
        errors=errors,
        empirical_risk=errors / len(selected) if selected else 0.0,
        risk_upper_bound=upper,
        population_gain_after_cost=population_gain,
        selected_mean_gain_after_cost=selected_mean,
        alpha=alpha,
        delta=delta,
        certified=bool(selected) and upper <= alpha,
    )


def _prefix_threshold(scores: list[float], order: list[int], count: int) -> float:
    selected_score = scores[order[count - 1]]
    if count == len(order):
        return selected_score - 1e-12
    next_score = scores[order[count]]
    return (selected_score + next_score) / 2.0
