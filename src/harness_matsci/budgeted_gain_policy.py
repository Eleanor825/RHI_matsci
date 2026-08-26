from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .risk_control import binomial_upper_confidence


@dataclass(frozen=True)
class BudgetedGainPolicy:
    budget_fraction: float = 0.10
    minimum_score: float = 0.0
    scope: str = "global"

    def __post_init__(self) -> None:
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError("budget_fraction must lie in (0, 1]")
        if not math.isfinite(self.minimum_score):
            raise ValueError("minimum_score must be finite")
        if self.scope != "global":
            raise ValueError("strict scientific-gain policy currently requires global scope")

    def select(self, scores: Sequence[float]) -> tuple[int, ...]:
        if not scores:
            return ()
        if any(not math.isfinite(float(score)) for score in scores):
            raise ValueError("policy scores must be finite")
        budget = max(1, math.ceil(self.budget_fraction * len(scores)))
        eligible = [
            index
            for index, score in enumerate(scores)
            if float(score) > self.minimum_score
        ]
        eligible.sort(key=lambda index: (-float(scores[index]), index))
        return tuple(eligible[:budget])

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


def evaluate_budgeted_gain_policy(
    policy: BudgetedGainPolicy,
    scores: Sequence[float],
    realized_gains: Sequence[float],
) -> dict[str, float | int]:
    if len(scores) != len(realized_gains):
        raise ValueError("scores and realized gains must align")
    selected = policy.select(scores)
    gains = [float(realized_gains[index]) for index in selected]
    errors = sum(gain <= 0.0 for gain in gains)
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(scores) if scores else 0.0,
        "errors": errors,
        "selective_risk": errors / len(selected) if selected else 0.0,
        "selected_mean_net_gain": sum(gains) / len(gains) if gains else 0.0,
        "selected_total_net_gain": sum(gains),
        "population_net_gain": sum(gains) / len(scores) if scores else 0.0,
    }


def certify_budgeted_gain_policy(
    policy: BudgetedGainPolicy,
    scores: Sequence[float],
    realized_gains: Sequence[float],
    *,
    alpha: float,
    delta: float = 0.05,
) -> dict[str, float | int | bool | str | dict[str, float | str]]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    evaluation = evaluate_budgeted_gain_policy(policy, scores, realized_gains)
    count = int(evaluation["selected_count"])
    errors = int(evaluation["errors"])
    upper = binomial_upper_confidence(errors, count, delta) if count else 1.0
    return {
        "guarantee": "exact one-sided Clopper-Pearson bound for a pre-fixed policy",
        "policy": policy.to_json(),
        "alpha": alpha,
        "delta": delta,
        **evaluation,
        "risk_upper_bound": upper,
        "certified": bool(count and upper <= alpha),
    }
