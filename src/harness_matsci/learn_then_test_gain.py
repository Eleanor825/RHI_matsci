from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .risk_control import binomial_upper_confidence


@dataclass(frozen=True)
class FixedThresholdGainPolicy:
    threshold: float
    budget_fraction: float
    thinning_seed: int

    def __post_init__(self) -> None:
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError("budget_fraction must lie in (0, 1]")

    def eligible(self, scores: Sequence[float]) -> tuple[int, ...]:
        return tuple(
            index for index, score in enumerate(scores) if float(score) >= self.threshold
        )

    def select(
        self,
        scores: Sequence[float],
        record_ids: Sequence[str],
    ) -> tuple[int, ...]:
        if len(scores) != len(record_ids):
            raise ValueError("scores and record_ids must align")
        eligible = self.eligible(scores)
        budget = max(1, math.ceil(self.budget_fraction * len(scores)))
        if len(eligible) <= budget:
            return eligible
        ranked = sorted(
            eligible,
            key=lambda index: (
                _stable_uniform_key(record_ids[index], self.thinning_seed),
                record_ids[index],
            ),
        )
        return tuple(sorted(ranked[:budget]))

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdCandidateEvidence:
    threshold: float
    selected_count: int
    errors: int
    empirical_risk: float
    simultaneous_risk_upper_bound: float
    selected_mean_gain: float
    selected_total_gain: float
    population_gain: float
    certified: bool

    def to_json(self) -> dict[str, float | int | bool]:
        return asdict(self)


def fit_learn_then_test_gain_policy(
    *,
    candidate_thresholds: Sequence[float],
    feedback_scores: Sequence[float],
    feedback_gains: Sequence[float],
    feedback_record_ids: Sequence[str],
    alpha: float,
    delta: float,
    budget_fraction: float,
    thinning_seed: int,
) -> tuple[FixedThresholdGainPolicy | None, list[ThresholdCandidateEvidence]]:
    thresholds = tuple(sorted({float(value) for value in candidate_thresholds}))
    if not thresholds:
        raise ValueError("candidate_thresholds must be non-empty")
    if not (
        len(feedback_scores) == len(feedback_gains) == len(feedback_record_ids)
        and feedback_scores
    ):
        raise ValueError("feedback arrays must be non-empty and aligned")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must lie in (0, 1)")
    family_delta = delta / len(thresholds)
    evidence = []
    for threshold in thresholds:
        policy = FixedThresholdGainPolicy(
            threshold=threshold,
            budget_fraction=budget_fraction,
            thinning_seed=thinning_seed,
        )
        selected = policy.select(feedback_scores, feedback_record_ids)
        gains = [float(feedback_gains[index]) for index in selected]
        errors = sum(gain <= 0.0 for gain in gains)
        upper = (
            binomial_upper_confidence(errors, len(selected), family_delta)
            if selected
            else 1.0
        )
        total = sum(gains)
        evidence.append(
            ThresholdCandidateEvidence(
                threshold=threshold,
                selected_count=len(selected),
                errors=errors,
                empirical_risk=errors / len(selected) if selected else 0.0,
                simultaneous_risk_upper_bound=upper,
                selected_mean_gain=total / len(selected) if selected else 0.0,
                selected_total_gain=total,
                population_gain=total / len(feedback_scores),
                certified=bool(selected) and upper <= alpha and total > 0.0,
            )
        )
    valid = [row for row in evidence if row.certified]
    if not valid:
        return None, evidence
    chosen = max(
        valid,
        key=lambda row: (
            row.population_gain,
            row.selected_total_gain,
            -row.simultaneous_risk_upper_bound,
            row.threshold,
        ),
    )
    return (
        FixedThresholdGainPolicy(
            threshold=chosen.threshold,
            budget_fraction=budget_fraction,
            thinning_seed=thinning_seed,
        ),
        evidence,
    )


def evaluate_fixed_threshold_gain_policy(
    policy: FixedThresholdGainPolicy,
    scores: Sequence[float],
    gains: Sequence[float],
    record_ids: Sequence[str],
    *,
    alpha: float,
    delta: float,
) -> dict[str, object]:
    if not (len(scores) == len(gains) == len(record_ids)):
        raise ValueError("evaluation arrays must align")
    selected = policy.select(scores, record_ids)
    selected_gains = [float(gains[index]) for index in selected]
    errors = sum(gain <= 0.0 for gain in selected_gains)
    upper = (
        binomial_upper_confidence(errors, len(selected), delta) if selected else 1.0
    )
    total = sum(selected_gains)
    return {
        "policy": policy.to_json(),
        "guarantee": (
            "one-sided Clopper-Pearson certificate for a frozen action-wise "
            "eligibility threshold under i.i.d. deployment actions"
        ),
        "alpha": alpha,
        "delta": delta,
        "eligible_count": len(policy.eligible(scores)),
        "selected_count": len(selected),
        "coverage": len(selected) / len(scores) if scores else 0.0,
        "errors": errors,
        "empirical_risk": errors / len(selected) if selected else 0.0,
        "risk_upper_bound": upper,
        "selected_mean_gain": total / len(selected) if selected else 0.0,
        "selected_total_gain": total,
        "population_gain": total / len(scores) if scores else 0.0,
        "certified": bool(selected) and upper <= alpha and total > 0.0,
    }


def _stable_uniform_key(record_id: str, seed: int) -> int:
    payload = f"{seed}::{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
