from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


FORBIDDEN_PRE_ACTION_KEYS = frozenset(
    {
        "hidden_outcome",
        "hidden_outcome_for_evaluation_only",
        "observed_success",
        "scientific_utility",
        "failure_harm",
        "reward",
        "label",
        "outcome_success",
    }
)


@dataclass(frozen=True)
class MatXBotPreActionEvent:
    event_id: str
    run_id: str
    parent_step_id: str | None
    system: str
    stage: str
    visible_context: dict[str, Any]
    proposed_action: dict[str, Any]
    available_tools: tuple[str, ...]
    model_replica_readouts: tuple[dict[str, Any], ...]
    evidence_replica_outputs: tuple[dict[str, Any], ...]
    estimated_cost: float
    reversibility: float
    timestamp: str

    def __post_init__(self) -> None:
        if not self.event_id or not self.run_id or not self.stage:
            raise ValueError("pre-action event requires event_id, run_id, and stage")
        if not math.isfinite(self.estimated_cost) or self.estimated_cost < 0.0:
            raise ValueError("estimated_cost must be finite and non-negative")
        if not math.isfinite(self.reversibility) or not 0.0 <= self.reversibility <= 1.0:
            raise ValueError("reversibility must lie in [0, 1]")
        forbidden = _find_forbidden_keys(
            {
                "visible_context": self.visible_context,
                "proposed_action": self.proposed_action,
                "model_replica_readouts": self.model_replica_readouts,
                "evidence_replica_outputs": self.evidence_replica_outputs,
            }
        )
        if forbidden:
            raise ValueError(f"pre-action event contains hidden outcome keys: {sorted(forbidden)}")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatXBotOutcomeEvent:
    event_id: str
    pre_action_event_id: str
    tool_results: tuple[dict[str, Any], ...]
    execution_cost: float
    observed_success: int
    scientific_utility: float
    failure_harm: float
    next_stage: str | None
    timestamp: str

    def __post_init__(self) -> None:
        if self.observed_success not in (0, 1):
            raise ValueError("observed_success must be binary")
        for name, value in (
            ("execution_cost", self.execution_cost),
            ("scientific_utility", self.scientific_utility),
            ("failure_harm", self.failure_harm),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def pair_matxbot_events(
    pre_actions: list[MatXBotPreActionEvent],
    outcomes: list[MatXBotOutcomeEvent],
) -> list[dict[str, Any]]:
    pre_by_id = {event.event_id: event for event in pre_actions}
    if len(pre_by_id) != len(pre_actions):
        raise ValueError("duplicate pre-action event IDs")
    outcome_by_pre: dict[str, MatXBotOutcomeEvent] = {}
    for outcome in outcomes:
        if outcome.pre_action_event_id not in pre_by_id:
            raise ValueError(
                f"outcome references unknown pre-action event {outcome.pre_action_event_id}"
            )
        if outcome.pre_action_event_id in outcome_by_pre:
            raise ValueError(
                f"multiple outcomes reference pre-action event {outcome.pre_action_event_id}"
            )
        outcome_by_pre[outcome.pre_action_event_id] = outcome
    return [
        {
            "trajectory_id": pre.run_id,
            "step_id": pre.event_id,
            "parent_step_id": pre.parent_step_id,
            "pre_action": pre.to_json(),
            "outcome_for_training_and_evaluation_only": (
                outcome_by_pre[pre.event_id].to_json()
                if pre.event_id in outcome_by_pre
                else None
            ),
            "outcome_available": pre.event_id in outcome_by_pre,
        }
        for pre in pre_actions
    ]


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_PRE_ACTION_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found
