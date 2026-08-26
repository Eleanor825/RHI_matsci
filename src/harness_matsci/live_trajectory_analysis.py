from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import fmean, pvariance

from .matbot_trajectory import MatBotTrajectory


NUMERIC_OUTPUT_FIELDS = (
    "p_success",
    "expected_scientific_utility",
    "reported_internal_uncertainty",
    "reported_external_uncertainty",
    "expected_value_of_evidence",
    "irreversible_risk",
    "expected_net_scientific_gain",
)


@dataclass(frozen=True)
class LiveTrajectoryAnalysis:
    trajectory_count: int
    model_count: int
    complete_count: int
    total_tool_calls: int
    total_tool_errors: int
    tool_error_rate: float
    numeric_outputs: dict[str, dict[str, float | list[float]]]
    action_similarity: dict[str, float | int | None]
    within_model_action_similarity: dict[str, dict[str, float | int | None]]
    within_model_numeric_dispersion: dict[str, dict[str, dict[str, float | list[float]]]]
    reported_internal_uncertainty_is_calibrated: bool
    within_model_replication_available: bool
    outcome_validation_available: bool

    def to_json(self) -> dict[str, object]:
        return {
            "trajectory_count": self.trajectory_count,
            "model_count": self.model_count,
            "complete_count": self.complete_count,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "tool_error_rate": self.tool_error_rate,
            "numeric_outputs": self.numeric_outputs,
            "action_similarity": self.action_similarity,
            "within_model_action_similarity": self.within_model_action_similarity,
            "within_model_numeric_dispersion": self.within_model_numeric_dispersion,
            "reported_internal_uncertainty_is_calibrated": (
                self.reported_internal_uncertainty_is_calibrated
            ),
            "within_model_replication_available": self.within_model_replication_available,
            "outcome_validation_available": self.outcome_validation_available,
            "interpretation": {
                "between_model_dispersion": (
                    "Descriptive disagreement across different model backends; it is not "
                    "an estimate of within-model internal variance."
                ),
                "reported_internal_uncertainty": (
                    "A model self-report until repeated sampling and observed outcomes are "
                    "available for calibration."
                ),
                "reported_external_uncertainty": (
                    "A model self-report until linked to controlled evidence perturbations or "
                    "source interventions."
                ),
            },
        }


def analyze_live_trajectories(
    trajectories: list[MatBotTrajectory],
) -> LiveTrajectoryAnalysis:
    if not trajectories:
        raise ValueError("at least one live trajectory is required")
    if any(not trajectory.is_live_runtime for trajectory in trajectories):
        raise ValueError("analysis accepts live_runtime trajectories only")
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    if any(len(trajectory.steps) != 1 for trajectory in trajectories):
        raise ValueError("current cross-model analysis expects one-step trajectories")
    models = [step.agent_backend or "unknown" for step in steps]
    numeric_outputs = {}
    for field in NUMERIC_OUTPUT_FIELDS:
        values = [float(step.uncertainty_output[field]) for step in steps]
        numeric_outputs[field] = {
            "values": values,
            "mean": fmean(values),
            "population_variance": pvariance(values),
            "range": max(values) - min(values),
        }
    action_texts = [step.action_text or "" for step in steps]
    character_jaccards = _pairwise_jaccards(action_texts, _character_bigrams)
    token_jaccards = _pairwise_jaccards(action_texts, _tokens)
    total_tool_calls = sum(len(step.tool_trace) for step in steps)
    total_tool_errors = sum(
        bool(tool.get("is_error")) for step in steps for tool in step.tool_trace
    )
    model_counts = {model: models.count(model) for model in set(models)}
    within_model_numeric_dispersion = {}
    within_model_action_similarity = {}
    for model in sorted(model_counts):
        model_steps = [step for step in steps if (step.agent_backend or "unknown") == model]
        if len(model_steps) < 2:
            continue
        model_actions = [step.action_text or "" for step in model_steps]
        model_character_jaccards = _pairwise_jaccards(model_actions, _character_bigrams)
        model_token_jaccards = _pairwise_jaccards(model_actions, _tokens)
        within_model_action_similarity[model] = {
            "pair_count": len(model_character_jaccards),
            "character_bigram_mean_jaccard": _optional_mean(model_character_jaccards),
            "character_bigram_min_jaccard": min(model_character_jaccards),
            "token_mean_jaccard": _optional_mean(model_token_jaccards),
            "token_min_jaccard": min(model_token_jaccards),
        }
        within_model_numeric_dispersion[model] = {
            field: _summarize(
                [float(step.uncertainty_output[field]) for step in model_steps]
            )
            for field in NUMERIC_OUTPUT_FIELDS
        }
    complete_count = sum(trajectory.complete for trajectory in trajectories)
    return LiveTrajectoryAnalysis(
        trajectory_count=len(trajectories),
        model_count=len(set(models)),
        complete_count=complete_count,
        total_tool_calls=total_tool_calls,
        total_tool_errors=total_tool_errors,
        tool_error_rate=total_tool_errors / total_tool_calls if total_tool_calls else math.nan,
        numeric_outputs=numeric_outputs,
        action_similarity={
            "pair_count": len(character_jaccards),
            "character_bigram_mean_jaccard": _optional_mean(character_jaccards),
            "character_bigram_min_jaccard": min(character_jaccards) if character_jaccards else None,
            "token_mean_jaccard": _optional_mean(token_jaccards),
            "token_min_jaccard": min(token_jaccards) if token_jaccards else None,
        },
        within_model_action_similarity=within_model_action_similarity,
        within_model_numeric_dispersion=within_model_numeric_dispersion,
        reported_internal_uncertainty_is_calibrated=False,
        within_model_replication_available=any(count >= 2 for count in model_counts.values()),
        outcome_validation_available=complete_count == len(trajectories),
    )


def _character_bigrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.casefold())
    return {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+\d*[a-z\d]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _pairwise_jaccards(texts: list[str], transform) -> list[float]:
    return [
        _jaccard(transform(texts[left]), transform(texts[right]))
        for left in range(len(texts))
        for right in range(left + 1, len(texts))
    ]


def _optional_mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": fmean(values),
        "population_variance": pvariance(values),
        "range": max(values) - min(values),
    }
