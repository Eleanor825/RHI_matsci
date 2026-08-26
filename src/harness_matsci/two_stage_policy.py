from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .hurdle_gain_calibration import HurdleGainPrediction
from .risk_control import binomial_upper_confidence


@dataclass(frozen=True)
class TwoStagePolicy:
    acquisition_threshold: float
    target_acquisition_coverage: float
    base_execution_threshold: float
    target_base_execution_coverage: float
    tool_execution_threshold: float
    target_tool_execution_coverage: float
    feedback_population_gain: float
    feedback_selective_risk: float
    selection_mode: str = "absolute_threshold"

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class TwoStageEvaluation:
    acquired_count: int
    acquisition_coverage: float
    executed_count: int
    execution_coverage: float
    errors: int
    selective_risk: float
    risk_upper_bound: float
    population_gain_after_cost: float
    alpha: float
    delta: float
    certified: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionThreshold:
    threshold: float
    target_coverage: float
    population_gain: float
    selective_risk: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


def select_execution_threshold(
    predictions: list[HurdleGainPrediction],
    labels: list[int],
    gains: list[float],
    *,
    coverages: tuple[float, ...],
    max_empirical_risk: float = 0.20,
) -> ExecutionThreshold:
    if not predictions or len(predictions) != len(labels) or len(labels) != len(gains):
        raise ValueError("execution-threshold inputs must be non-empty and aligned")
    scores = [prediction.tail_risk_score for prediction in predictions]
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    best = ExecutionThreshold(math.inf, 0.0, 0.0, 0.0)
    for coverage, threshold in _coverage_thresholds(scores, order, (0.0, *coverages)):
        selected = [index for index, score in enumerate(scores) if score >= threshold]
        risk = (
            sum(1 - int(labels[index]) for index in selected) / len(selected)
            if selected
            else 0.0
        )
        if risk > max_empirical_risk:
            continue
        population_gain = sum(gains[index] for index in selected) / len(gains)
        candidate = ExecutionThreshold(
            threshold=threshold,
            target_coverage=coverage,
            population_gain=population_gain,
            selective_risk=risk,
        )
        if (
            candidate.population_gain,
            len(selected),
            -candidate.target_coverage,
        ) > (
            best.population_gain,
            _threshold_selected_count(best.threshold, scores),
            -best.target_coverage,
        ):
            best = candidate
    return best


def select_two_stage_policy(
    acquisition_scores: list[float],
    base_predictions: list[HurdleGainPrediction],
    tool_predictions: list[HurdleGainPrediction],
    labels: list[int],
    gains: list[float],
    *,
    tool_cost: float,
    coverages: tuple[float, ...],
    execution_coverages: tuple[float, ...] | None = None,
    base_execution_coverages: tuple[float, ...] | None = None,
    tool_execution_coverages: tuple[float, ...] | None = None,
    max_empirical_risk: float = 0.20,
    selection_mode: str = "absolute_threshold",
) -> TwoStagePolicy:
    _validate(acquisition_scores, base_predictions, tool_predictions, labels, gains)
    if selection_mode not in ("absolute_threshold", "rank_budget"):
        raise ValueError(f"unknown selection mode {selection_mode}")
    acquisition_candidates = [0.0, *coverages]
    shared_execution_coverages = execution_coverages or coverages
    base_execution_candidates = [
        0.0,
        *(base_execution_coverages or shared_execution_coverages),
    ]
    tool_execution_candidates = [
        0.0,
        *(tool_execution_coverages or shared_execution_coverages),
    ]
    best = None
    acquisition_order = sorted(
        range(len(acquisition_scores)),
        key=lambda index: (-acquisition_scores[index], index),
    )
    base_scores = [prediction.tail_risk_score for prediction in base_predictions]
    tool_scores = [prediction.tail_risk_score for prediction in tool_predictions]
    base_order = sorted(
        range(len(base_scores)), key=lambda index: (-base_scores[index], index)
    )
    tool_order = sorted(
        range(len(tool_scores)), key=lambda index: (-tool_scores[index], index)
    )
    acquisition_thresholds = _coverage_thresholds(
        acquisition_scores, acquisition_order, acquisition_candidates
    )
    base_thresholds = _coverage_thresholds(
        base_scores, base_order, base_execution_candidates
    )
    tool_thresholds = _coverage_thresholds(
        tool_scores, tool_order, tool_execution_candidates
    )
    for acquisition_coverage, acquisition_threshold in acquisition_thresholds:
        for base_coverage, base_threshold in base_thresholds:
            for tool_coverage, tool_threshold in tool_thresholds:
                provisional = TwoStagePolicy(
                    acquisition_threshold=acquisition_threshold,
                    target_acquisition_coverage=acquisition_coverage,
                    base_execution_threshold=base_threshold,
                    target_base_execution_coverage=base_coverage,
                    tool_execution_threshold=tool_threshold,
                    target_tool_execution_coverage=tool_coverage,
                    feedback_population_gain=0.0,
                    feedback_selective_risk=0.0,
                    selection_mode=selection_mode,
                )
                evaluation = evaluate_two_stage_policy(
                    provisional,
                    acquisition_scores,
                    base_predictions,
                    tool_predictions,
                    labels,
                    gains,
                    tool_cost=tool_cost,
                    alpha=1.0 - 1e-12,
                    delta=0.5,
                )
                if evaluation.selective_risk > max_empirical_risk:
                    continue
                candidate = TwoStagePolicy(
                    acquisition_threshold=acquisition_threshold,
                    target_acquisition_coverage=acquisition_coverage,
                    base_execution_threshold=base_threshold,
                    target_base_execution_coverage=base_coverage,
                    tool_execution_threshold=tool_threshold,
                    target_tool_execution_coverage=tool_coverage,
                    feedback_population_gain=evaluation.population_gain_after_cost,
                    feedback_selective_risk=evaluation.selective_risk,
                    selection_mode=selection_mode,
                )
                if best is None or (
                    candidate.feedback_population_gain,
                    evaluation.executed_count,
                    -candidate.target_acquisition_coverage,
                ) > (
                    best.feedback_population_gain,
                    _executed_count(best, acquisition_scores, base_predictions, tool_predictions),
                    -best.target_acquisition_coverage,
                ):
                    best = candidate
    if best is None or best.feedback_population_gain <= 0.0:
        return TwoStagePolicy(
            math.inf,
            0.0,
            math.inf,
            0.0,
            math.inf,
            0.0,
            0.0,
            0.0,
            selection_mode,
        )
    return best


def evaluate_two_stage_policy(
    policy: TwoStagePolicy,
    acquisition_scores: list[float],
    base_predictions: list[HurdleGainPrediction],
    tool_predictions: list[HurdleGainPrediction],
    labels: list[int],
    gains: list[float],
    *,
    tool_cost: float,
    alpha: float,
    delta: float,
) -> TwoStageEvaluation:
    _validate(acquisition_scores, base_predictions, tool_predictions, labels, gains)
    acquired, executed = _policy_masks(
        policy, acquisition_scores, base_predictions, tool_predictions
    )
    executed_indices = [index for index, execute in enumerate(executed) if execute]
    errors = sum(1 - int(labels[index]) for index in executed_indices)
    upper = binomial_upper_confidence(errors, len(executed_indices), delta)
    net_gain = sum(
        (gains[index] if executed[index] else 0.0)
        - (tool_cost if acquired[index] else 0.0)
        for index in range(len(gains))
    )
    return TwoStageEvaluation(
        acquired_count=sum(acquired),
        acquisition_coverage=sum(acquired) / len(acquired),
        executed_count=len(executed_indices),
        execution_coverage=len(executed_indices) / len(executed),
        errors=errors,
        selective_risk=errors / len(executed_indices) if executed_indices else 0.0,
        risk_upper_bound=upper,
        population_gain_after_cost=net_gain / len(gains),
        alpha=alpha,
        delta=delta,
        certified=bool(executed_indices) and upper <= alpha,
    )


def _validate(scores, base_predictions, tool_predictions, labels, gains) -> None:
    count = len(scores)
    if count == 0 or any(
        len(values) != count
        for values in (base_predictions, tool_predictions, labels, gains)
    ):
        raise ValueError("two-stage policy inputs must be non-empty and aligned")


def _prefix_threshold(scores: list[float], order: list[int], count: int) -> float:
    selected_score = scores[order[count - 1]]
    if count == len(order):
        return selected_score - 1e-12
    return (selected_score + scores[order[count]]) / 2.0


def _coverage_thresholds(scores, order, coverages):
    output = []
    for coverage in coverages:
        if coverage == 0.0:
            output.append((coverage, math.inf))
            continue
        count = max(1, min(len(order), math.ceil(len(order) * coverage)))
        output.append((coverage, _prefix_threshold(scores, order, count)))
    return output


def _executed_count(policy, acquisition_scores, base_predictions, tool_predictions):
    _, executed = _policy_masks(
        policy, acquisition_scores, base_predictions, tool_predictions
    )
    return sum(executed)


def _policy_masks(policy, acquisition_scores, base_predictions, tool_predictions):
    if policy.selection_mode == "absolute_threshold":
        acquired = [score >= policy.acquisition_threshold for score in acquisition_scores]
        executed = [
            (
                tool_prediction.tail_risk_score >= policy.tool_execution_threshold
                if acquire
                else base_prediction.tail_risk_score >= policy.base_execution_threshold
            )
            for acquire, base_prediction, tool_prediction in zip(
                acquired, base_predictions, tool_predictions
            )
        ]
        return acquired, executed
    if policy.selection_mode != "rank_budget":
        raise ValueError(f"unknown selection mode {policy.selection_mode}")
    acquired = _rank_budget_mask(
        acquisition_scores,
        policy.target_acquisition_coverage,
    )
    base_eligible = [not value for value in acquired]
    tool_eligible = acquired
    base_selected = _rank_budget_mask(
        [prediction.tail_risk_score for prediction in base_predictions],
        policy.target_base_execution_coverage,
        eligible=base_eligible,
    )
    tool_selected = _rank_budget_mask(
        [prediction.tail_risk_score for prediction in tool_predictions],
        policy.target_tool_execution_coverage,
        eligible=tool_eligible,
    )
    executed = [
        tool_selected[index] if acquired[index] else base_selected[index]
        for index in range(len(acquired))
    ]
    return acquired, executed


def _rank_budget_mask(scores, target_coverage, *, eligible=None):
    count = len(scores)
    eligible_mask = [True] * count if eligible is None else list(eligible)
    if len(eligible_mask) != count:
        raise ValueError("rank-budget eligibility must align with scores")
    target_count = min(
        sum(eligible_mask),
        max(0, math.ceil(count * target_coverage)),
    )
    selected = [False] * count
    if target_count == 0:
        return selected
    order = sorted(
        (index for index in range(count) if eligible_mask[index]),
        key=lambda index: (-scores[index], index),
    )
    for index in order[:target_count]:
        selected[index] = True
    return selected


def _threshold_selected_count(threshold, scores):
    return sum(score >= threshold for score in scores)
