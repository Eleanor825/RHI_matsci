from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .risk_control import binomial_upper_confidence


@dataclass(frozen=True)
class TaskBalancedExecutionPolicy:
    worthiness_threshold: float
    budget_fraction: float

    def select(
        self,
        *,
        decision_scores: Sequence[float],
        worthiness_probabilities: Sequence[float],
        benchmarks: Sequence[str],
        record_ids: Sequence[str],
    ) -> tuple[int, ...]:
        sizes = {
            len(decision_scores),
            len(worthiness_probabilities),
            len(benchmarks),
            len(record_ids),
        }
        if len(sizes) != 1 or not decision_scores:
            raise ValueError("execution-policy arrays must be non-empty and aligned")
        selected = []
        for benchmark in sorted(set(benchmarks)):
            task_indices = [
                index
                for index, task in enumerate(benchmarks)
                if task == benchmark
            ]
            budget = max(1, math.ceil(self.budget_fraction * len(task_indices)))
            eligible = [
                index
                for index in task_indices
                if float(decision_scores[index]) > 0.0
                and float(worthiness_probabilities[index])
                >= self.worthiness_threshold
            ]
            eligible.sort(
                key=lambda index: (-float(decision_scores[index]), record_ids[index])
            )
            selected.extend(eligible[:budget])
        return tuple(sorted(selected))

    def to_json(self) -> dict[str, float | str]:
        return {
            **asdict(self),
            "type": "task_balanced_calibrated_worthiness_threshold",
        }


@dataclass(frozen=True)
class ExecutionThresholdEvidence:
    threshold: float
    selected_count: int
    errors: int
    empirical_risk: float
    simultaneous_risk_upper_bound: float
    selected_total_gain: float
    selected_mean_gain: float
    certified: bool

    def to_json(self) -> dict[str, float | int | bool]:
        return asdict(self)


def fit_task_balanced_execution_policy(
    *,
    candidate_thresholds: Sequence[float],
    decision_scores: Sequence[float],
    worthiness_probabilities: Sequence[float],
    realized_gains: Sequence[float],
    benchmarks: Sequence[str],
    record_ids: Sequence[str],
    budget_fraction: float,
    alpha: float,
    delta: float,
) -> tuple[TaskBalancedExecutionPolicy | None, list[ExecutionThresholdEvidence]]:
    thresholds = tuple(sorted({float(value) for value in candidate_thresholds}))
    if not thresholds:
        raise ValueError("execution-policy thresholds must be non-empty")
    sizes = {
        len(decision_scores),
        len(worthiness_probabilities),
        len(realized_gains),
        len(benchmarks),
        len(record_ids),
    }
    if len(sizes) != 1 or not decision_scores:
        raise ValueError("execution-policy feedback arrays must align")
    family_delta = delta / len(thresholds)
    evidence = []
    for threshold in thresholds:
        policy = TaskBalancedExecutionPolicy(threshold, budget_fraction)
        selected = policy.select(
            decision_scores=decision_scores,
            worthiness_probabilities=worthiness_probabilities,
            benchmarks=benchmarks,
            record_ids=record_ids,
        )
        gains = [float(realized_gains[index]) for index in selected]
        errors = sum(gain <= 0.0 for gain in gains)
        upper = (
            binomial_upper_confidence(errors, len(gains), family_delta)
            if gains
            else 1.0
        )
        total = sum(gains)
        evidence.append(
            ExecutionThresholdEvidence(
                threshold=threshold,
                selected_count=len(gains),
                errors=errors,
                empirical_risk=errors / len(gains) if gains else 0.0,
                simultaneous_risk_upper_bound=upper,
                selected_total_gain=total,
                selected_mean_gain=total / len(gains) if gains else 0.0,
                certified=bool(gains) and upper <= alpha and total > 0.0,
            )
        )
    valid = [row for row in evidence if row.certified]
    if not valid:
        return None, evidence
    chosen = max(
        valid,
        key=lambda row: (
            row.selected_total_gain,
            row.selected_count,
            -row.simultaneous_risk_upper_bound,
            -row.threshold,
        ),
    )
    return TaskBalancedExecutionPolicy(chosen.threshold, budget_fraction), evidence
