from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .factorized_gain_harness import FactorizedGainPrediction
from .matbot_trajectory import SCHEMA_VERSION, assemble_matbot_trajectories


def strict_gain_pre_action_event(
    *,
    event_id: str,
    run_id: str,
    trajectory_id: str,
    step_id: int,
    timestamp: str,
    system: str,
    stage: str,
    action_id: str,
    action_type: str,
    visible_context: str,
    evidence: list[str],
    internal_signals: dict[str, float],
    external_signals: dict[str, float],
    normalized_cost: float,
    reversibility: float,
    prediction: FactorizedGainPrediction,
    conformal_lower_gain: float,
    capture_mode: str = "live_runtime",
    agent_backend: str | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
    action_text: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    values = {
        "positive_gain_probability": prediction.action_worthiness,
        "expected_strict_net_gain": prediction.expected_net_gain,
        "p_success": prediction.success_probability,
        "conditional_utility_mean": prediction.conditional_utility_mean,
        "probability_internal_variance": prediction.probability_internal_variance,
        "probability_external_variance": prediction.probability_external_variance,
        "probability_interaction_variance": prediction.probability_interaction_variance,
        "gain_internal_variance": prediction.gain_internal_variance,
        "gain_external_variance": prediction.gain_external_variance,
        "gain_interaction_variance": prediction.gain_interaction_variance,
        "aleatoric_gain_variance": prediction.aleatoric_gain_variance,
        "total_gain_variance": prediction.total_gain_variance,
        "conformal_lower_gain": conformal_lower_gain,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("strict gain uncertainty output must be finite")
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "pre_action",
        "event_id": event_id,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "step_id": step_id,
        "timestamp": timestamp,
        "capture_mode": capture_mode,
        "system": system,
        "stage": stage,
        "action_id": action_id,
        "action_type": action_type,
        "visible_context": visible_context,
        "evidence": list(evidence),
        "internal_signals": _numeric(internal_signals),
        "external_signals": _numeric(external_signals),
        "cost_signals": {
            "normalized_cost": float(normalized_cost),
            "reversibility": float(reversibility),
        },
        "uncertainty_output": values,
        "hidden_outcome_exposed": False,
        "agent_backend": agent_backend,
        "tool_trace": list(tool_trace or ()),
        "action_text": action_text,
        "rationale": rationale,
    }
    if step_id == 0:
        assemble_matbot_trajectories([event])
    return event


def strict_gain_post_outcome_event(
    *,
    event_id: str,
    parent_event_id: str,
    run_id: str,
    trajectory_id: str,
    step_id: int,
    timestamp: str,
    observed_success: int,
    normalized_utility: float,
    realized_cost: float,
    failure_harm: float,
    terminal: bool,
    capture_mode: str = "live_runtime",
    cost_weight: float = 0.15,
    failure_harm_weight: float = 0.25,
) -> dict[str, Any]:
    if observed_success not in (0, 1):
        raise ValueError("observed success must be binary")
    for name, value in (
        ("normalized_utility", normalized_utility),
        ("realized_cost", realized_cost),
        ("failure_harm", failure_harm),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    gain = (
        observed_success * float(normalized_utility)
        - cost_weight * float(realized_cost)
        - failure_harm_weight * (1 - observed_success) * float(failure_harm)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": "post_outcome",
        "event_id": event_id,
        "parent_event_id": parent_event_id,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "step_id": step_id,
        "timestamp": timestamp,
        "capture_mode": capture_mode,
        "outcome_status": "positive_gain" if gain > 0.0 else "non_positive_gain",
        "observed_outcome": {
            "observed_success": observed_success,
            "normalized_scientific_utility": float(normalized_utility),
            "failure_harm": float(failure_harm),
            "strict_realized_net_gain": gain,
        },
        "realized_cost": float(realized_cost),
        "terminal": bool(terminal),
    }


def append_runtime_event(path: str | Path, event: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _numeric(values):
    output = {str(key): float(value) for key, value in values.items()}
    if any(not math.isfinite(value) for value in output.values()):
        raise ValueError("runtime signal maps must be finite")
    return output
