from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallEvent:
    event_id: str
    tool_name: str
    tool_type: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelReadoutEvent:
    event_id: str
    model: str
    evidence_view_id: str
    p_positive_gain: float
    expected_gain: float
    p_success: float
    assessment_confidence: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificActionTrajectory:
    trajectory_id: str
    source: str
    online: bool
    benchmark: str
    record_id: str
    visible_state: dict[str, Any]
    proposed_action: dict[str, Any]
    tool_calls: tuple[ToolCallEvent, ...]
    model_readouts: tuple[ModelReadoutEvent, ...]
    harness_output: dict[str, Any]
    hidden_outcome_for_evaluation_only: dict[str, Any]

    def __post_init__(self) -> None:
        if "selected_route" in self.harness_output or "route" in self.harness_output:
            raise ValueError("uncertainty harness output must not contain a route")
        if "action_worthiness" not in self.harness_output:
            raise ValueError("uncertainty harness output requires scalar action_worthiness")

    def to_json(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "source": self.source,
            "online": self.online,
            "benchmark": self.benchmark,
            "record_id": self.record_id,
            "visible_state": dict(self.visible_state),
            "proposed_action": dict(self.proposed_action),
            "tool_calls": [event.to_json() for event in self.tool_calls],
            "model_readouts": [event.to_json() for event in self.model_readouts],
            "harness_output": dict(self.harness_output),
            "hidden_outcome_for_evaluation_only": dict(self.hidden_outcome_for_evaluation_only),
        }
