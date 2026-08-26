from __future__ import annotations

import math
import itertools
from dataclasses import asdict, dataclass

from .hurdle_gain_calibration import HurdleGainPrediction
from .risk_control import binomial_upper_confidence


@dataclass(frozen=True)
class MultiToolPolicy:
    acquisition_threshold: float
    target_acquisition_coverage: float
    execution_threshold: float
    target_execution_coverage: float
    feedback_population_gain: float
    feedback_selective_risk: float
    selection_mode: str = "absolute_threshold"

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class MultiToolEvaluation:
    acquired_count: int
    acquisition_coverage: float
    executed_count: int
    execution_coverage: float
    errors: int
    selective_risk: float
    risk_upper_bound: float
    population_gain_after_cost: float
    selected_tool_counts: dict[str, int]
    executed_tool_counts: dict[str, int]
    alpha: float
    delta: float
    certified: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CertifiedToolContract:
    selected_tools: tuple[str, ...]
    policy: MultiToolPolicy
    acceptance: MultiToolEvaluation
    trace: tuple[dict[str, object], ...]
    familywise_delta: float

    def to_json(self) -> dict[str, object]:
        return {
            "selected_tools": list(self.selected_tools),
            "policy": self.policy.to_json(),
            "acceptance": self.acceptance.to_json(),
            "trace": list(self.trace),
            "familywise_delta": self.familywise_delta,
        }


def select_certified_tool_contract(
    feedback_scores: dict[str, list[float]],
    acceptance_scores: dict[str, list[float]],
    feedback_predictions: dict[str, list[HurdleGainPrediction]],
    acceptance_predictions: dict[str, list[HurdleGainPrediction]],
    feedback_labels: list[int],
    acceptance_labels: list[int],
    feedback_gains: list[float],
    acceptance_gains: list[float],
    *,
    tool_costs: dict[str, float],
    acquisition_coverages: tuple[float, ...],
    execution_coverages: tuple[float, ...],
    alpha: float,
    familywise_delta: float,
    baseline_tool: str | None = None,
    baseline_weight: float = 0.90,
    max_empirical_risk: float = 0.20,
    selection_mode: str = "absolute_threshold",
) -> CertifiedToolContract | None:
    tools = tuple(sorted(feedback_scores))
    subsets = tuple(
        subset
        for size in range(1, len(tools) + 1)
        for subset in itertools.combinations(tools, size)
    )
    if baseline_tool is not None and baseline_tool not in tools:
        raise ValueError(f"baseline tool {baseline_tool} is unavailable")
    if not 0.0 < baseline_weight < 1.0:
        raise ValueError("baseline weight must lie strictly between zero and one")
    baseline_subset = (baseline_tool,) if baseline_tool is not None else None
    expansion_count = len(subsets) - int(baseline_subset is not None)
    accepted = []
    trace = []
    for subset in subsets:
        if baseline_subset is None:
            candidate_delta = familywise_delta / len(subsets)
        elif subset == baseline_subset:
            candidate_delta = familywise_delta * baseline_weight
        else:
            candidate_delta = (
                familywise_delta * (1.0 - baseline_weight) / expansion_count
            )
        subset_costs = {tool: tool_costs[tool] for tool in subset}
        policy = select_multi_tool_policy(
            {tool: feedback_scores[tool] for tool in subset},
            {tool: feedback_predictions[tool] for tool in subset},
            feedback_labels,
            feedback_gains,
            tool_costs=subset_costs,
            acquisition_coverages=acquisition_coverages,
            execution_coverages=execution_coverages,
            max_empirical_risk=max_empirical_risk,
            selection_mode=selection_mode,
        )
        acceptance = evaluate_multi_tool_policy(
            policy,
            {tool: acceptance_scores[tool] for tool in subset},
            {tool: acceptance_predictions[tool] for tool in subset},
            acceptance_labels,
            acceptance_gains,
            tool_costs=subset_costs,
            alpha=alpha,
            delta=candidate_delta,
        )
        row = {
            "tools": list(subset),
            "feedback_population_gain": policy.feedback_population_gain,
            "acceptance_certified": acceptance.certified,
            "acceptance_risk_upper_bound": acceptance.risk_upper_bound,
            "acceptance_population_gain": acceptance.population_gain_after_cost,
            "candidate_delta": candidate_delta,
            "is_safety_baseline": subset == baseline_subset,
        }
        trace.append(row)
        if acceptance.certified:
            accepted.append((policy.feedback_population_gain, acceptance.executed_count, subset, policy, acceptance))
    if not accepted:
        return None
    _, _, subset, policy, acceptance = max(
        accepted,
        key=lambda row: (row[0], row[1], tuple(reversed(row[2]))),
    )
    return CertifiedToolContract(
        selected_tools=subset,
        policy=policy,
        acceptance=acceptance,
        trace=tuple(trace),
        familywise_delta=familywise_delta,
    )


def select_multi_tool_policy(
    tool_scores: dict[str, list[float]],
    tool_predictions: dict[str, list[HurdleGainPrediction]],
    labels: list[int],
    gains: list[float],
    *,
    tool_costs: dict[str, float],
    acquisition_coverages: tuple[float, ...],
    execution_coverages: tuple[float, ...],
    max_empirical_risk: float = 0.20,
    selection_mode: str = "absolute_threshold",
) -> MultiToolPolicy:
    _validate(tool_scores, tool_predictions, labels, gains, tool_costs)
    if selection_mode not in ("absolute_threshold", "rank_budget"):
        raise ValueError(f"unknown selection mode {selection_mode}")
    best_scores, selected_tools, selected_predictions = _best_tools(
        tool_scores, tool_predictions
    )
    acquisition_order = _descending_order(best_scores)
    execution_scores = [row.tail_risk_score for row in selected_predictions]
    execution_order = _descending_order(execution_scores)
    best_policy = None
    best_evaluation = None
    for acquisition_coverage, acquisition_threshold in _coverage_thresholds(
        best_scores, acquisition_order, (0.0, *acquisition_coverages)
    ):
        for execution_coverage, execution_threshold in _coverage_thresholds(
            execution_scores, execution_order, (0.0, *execution_coverages)
        ):
            candidate = MultiToolPolicy(
                acquisition_threshold=acquisition_threshold,
                target_acquisition_coverage=acquisition_coverage,
                execution_threshold=execution_threshold,
                target_execution_coverage=execution_coverage,
                feedback_population_gain=0.0,
                feedback_selective_risk=0.0,
                selection_mode=selection_mode,
            )
            evaluation = _evaluate_selected_tools(
                candidate,
                best_scores,
                selected_tools,
                selected_predictions,
                labels,
                gains,
                tool_costs=tool_costs,
                alpha=1.0 - 1e-12,
                delta=0.5,
            )
            if evaluation.selective_risk > max_empirical_risk:
                continue
            if best_evaluation is None or (
                evaluation.population_gain_after_cost,
                evaluation.executed_count,
                -candidate.target_acquisition_coverage,
            ) > (
                best_evaluation.population_gain_after_cost,
                best_evaluation.executed_count,
                -best_policy.target_acquisition_coverage,
            ):
                best_policy = candidate
                best_evaluation = evaluation
    if best_policy is None or best_evaluation.population_gain_after_cost <= 0.0:
        return MultiToolPolicy(
            math.inf,
            0.0,
            math.inf,
            0.0,
            0.0,
            0.0,
            selection_mode,
        )
    return MultiToolPolicy(
        acquisition_threshold=best_policy.acquisition_threshold,
        target_acquisition_coverage=best_policy.target_acquisition_coverage,
        execution_threshold=best_policy.execution_threshold,
        target_execution_coverage=best_policy.target_execution_coverage,
        feedback_population_gain=best_evaluation.population_gain_after_cost,
        feedback_selective_risk=best_evaluation.selective_risk,
        selection_mode=selection_mode,
    )


def evaluate_multi_tool_policy(
    policy: MultiToolPolicy,
    tool_scores: dict[str, list[float]],
    tool_predictions: dict[str, list[HurdleGainPrediction]],
    labels: list[int],
    gains: list[float],
    *,
    tool_costs: dict[str, float],
    alpha: float,
    delta: float,
) -> MultiToolEvaluation:
    _validate(tool_scores, tool_predictions, labels, gains, tool_costs)
    best_scores, selected_tools, selected_predictions = _best_tools(
        tool_scores, tool_predictions
    )
    return _evaluate_selected_tools(
        policy,
        best_scores,
        selected_tools,
        selected_predictions,
        labels,
        gains,
        tool_costs=tool_costs,
        alpha=alpha,
        delta=delta,
    )


def _evaluate_selected_tools(
    policy,
    best_scores,
    selected_tools,
    selected_predictions,
    labels,
    gains,
    *,
    tool_costs,
    alpha,
    delta,
):
    acquired = _selection_mask(
        best_scores,
        threshold=policy.acquisition_threshold,
        coverage=policy.target_acquisition_coverage,
        mode=policy.selection_mode,
    )
    execution_eligible = acquired
    executed = _selection_mask(
        [row.tail_risk_score for row in selected_predictions],
        threshold=policy.execution_threshold,
        coverage=policy.target_execution_coverage,
        mode=policy.selection_mode,
        eligible=execution_eligible,
    )
    executed_indices = [index for index, value in enumerate(executed) if value]
    errors = sum(1 - int(labels[index]) for index in executed_indices)
    upper = binomial_upper_confidence(errors, len(executed_indices), delta)
    net_gain = sum(
        (gains[index] if executed[index] else 0.0)
        - (tool_costs[selected_tools[index]] if acquired[index] else 0.0)
        for index in range(len(gains))
    )
    selected_tool_counts = {tool: 0 for tool in sorted(tool_costs)}
    executed_tool_counts = {tool: 0 for tool in sorted(tool_costs)}
    for index, tool in enumerate(selected_tools):
        if acquired[index]:
            selected_tool_counts[tool] += 1
        if executed[index]:
            executed_tool_counts[tool] += 1
    return MultiToolEvaluation(
        acquired_count=sum(acquired),
        acquisition_coverage=sum(acquired) / len(acquired),
        executed_count=len(executed_indices),
        execution_coverage=len(executed_indices) / len(executed),
        errors=errors,
        selective_risk=errors / len(executed_indices) if executed_indices else 0.0,
        risk_upper_bound=upper,
        population_gain_after_cost=net_gain / len(gains),
        selected_tool_counts=selected_tool_counts,
        executed_tool_counts=executed_tool_counts,
        alpha=alpha,
        delta=delta,
        certified=bool(executed_indices) and upper <= alpha,
    )


def _best_tools(tool_scores, tool_predictions):
    tools = sorted(tool_scores)
    count = len(next(iter(tool_scores.values())))
    selected_tools = []
    best_scores = []
    selected_predictions = []
    for index in range(count):
        tool = max(tools, key=lambda name: (tool_scores[name][index], name))
        selected_tools.append(tool)
        best_scores.append(float(tool_scores[tool][index]))
        selected_predictions.append(tool_predictions[tool][index])
    return best_scores, selected_tools, selected_predictions


def _selection_mask(scores, *, threshold, coverage, mode, eligible=None):
    eligible_mask = [True] * len(scores) if eligible is None else list(eligible)
    if mode == "absolute_threshold":
        return [
            bool(eligible_mask[index] and score >= threshold)
            for index, score in enumerate(scores)
        ]
    if mode != "rank_budget":
        raise ValueError(f"unknown selection mode {mode}")
    target_count = min(
        sum(eligible_mask),
        max(0, math.ceil(len(scores) * coverage)),
    )
    selected = [False] * len(scores)
    order = sorted(
        (index for index in range(len(scores)) if eligible_mask[index]),
        key=lambda index: (-scores[index], index),
    )
    for index in order[:target_count]:
        selected[index] = True
    return selected


def _coverage_thresholds(scores, order, coverages):
    output = []
    for coverage in coverages:
        if coverage == 0.0:
            output.append((coverage, math.inf))
            continue
        count = max(1, min(len(order), math.ceil(len(order) * coverage)))
        selected_score = scores[order[count - 1]]
        threshold = (
            selected_score - 1e-12
            if count == len(order)
            else (selected_score + scores[order[count]]) / 2.0
        )
        output.append((coverage, threshold))
    return output


def _descending_order(scores):
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _validate(tool_scores, tool_predictions, labels, gains, tool_costs):
    if not tool_scores or set(tool_scores) != set(tool_predictions):
        raise ValueError("tool score and prediction maps must be non-empty and aligned")
    if set(tool_scores) != set(tool_costs):
        raise ValueError("every tool requires one fixed cost")
    count = len(labels)
    if count == 0 or len(gains) != count:
        raise ValueError("labels and gains must be non-empty and aligned")
    if any(len(values) != count for values in tool_scores.values()):
        raise ValueError("tool scores must align with labels")
    if any(len(values) != count for values in tool_predictions.values()):
        raise ValueError("tool predictions must align with labels")
    if any(cost < 0.0 for cost in tool_costs.values()):
        raise ValueError("tool costs must be nonnegative")
