from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "matxbot-uncertainty-v1"
CAPTURE_MODES = {"live_runtime", "offline_replay", "synthetic_smoke"}


@dataclass(frozen=True)
class MatBotTrajectoryStep:
    step_id: int
    pre_action_event_id: str
    action_id: str
    action_type: str
    stage: str
    visible_context: str
    evidence: tuple[str, ...]
    internal_signals: dict[str, float]
    external_signals: dict[str, float]
    cost_signals: dict[str, float]
    uncertainty_output: dict[str, float]
    outcome_status: str | None
    observed_outcome: dict[str, Any] | None
    realized_cost: float | None
    terminal: bool
    agent_backend: str | None = None
    tool_trace: tuple[dict[str, Any], ...] = ()
    action_text: str | None = None
    rationale: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatBotTrajectory:
    trajectory_id: str
    run_id: str
    system: str
    capture_mode: str
    steps: tuple[MatBotTrajectoryStep, ...]
    complete: bool

    @property
    def is_live_runtime(self) -> bool:
        return self.capture_mode == "live_runtime"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_matbot_trajectories(path: str | Path) -> list[MatBotTrajectory]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return []
    if all("event_type" in row for row in rows):
        return assemble_matbot_trajectories(rows)
    if all("steps" in row and "trajectory_id" in row for row in rows):
        return [_trajectory_from_json(row) for row in rows]
    raise ValueError("trajectory file mixes or omits supported event/export formats")


def assemble_matbot_trajectories(events: list[dict[str, Any]]) -> list[MatBotTrajectory]:
    pre_actions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for event in events:
        _validate_common(event)
        if event["event_type"] == "pre_action":
            _validate_pre_action(event)
            if event["event_id"] in pre_actions:
                raise ValueError(f"duplicate pre-action event {event['event_id']}")
            pre_actions[event["event_id"]] = event
        elif event["event_type"] == "post_outcome":
            _validate_post_outcome(event)
            parent = event["parent_event_id"]
            if parent in outcomes:
                raise ValueError(f"duplicate post-outcome parent {parent}")
            outcomes[parent] = event
        else:
            raise ValueError(f"unknown event type {event['event_type']}")
    dangling = set(outcomes) - set(pre_actions)
    if dangling:
        raise ValueError(f"post-outcome events without pre-action parents: {sorted(dangling)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in pre_actions.values():
        grouped.setdefault(event["trajectory_id"], []).append(event)
    trajectories = []
    for trajectory_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["step_id"])
        if [row["step_id"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"trajectory {trajectory_id} has non-contiguous step ids")
        run_ids = {row["run_id"] for row in rows}
        systems = {row["system"] for row in rows}
        capture_modes = {row["capture_mode"] for row in rows}
        if len(run_ids) != 1 or len(systems) != 1 or len(capture_modes) != 1:
            raise ValueError(f"trajectory {trajectory_id} changes run, system, or capture mode")
        steps = []
        for row in rows:
            outcome = outcomes.get(row["event_id"])
            if outcome is not None and (
                outcome["trajectory_id"] != trajectory_id
                or outcome["step_id"] != row["step_id"]
                or outcome["capture_mode"] != row["capture_mode"]
            ):
                raise ValueError(f"outcome mismatch for event {row['event_id']}")
            steps.append(
                MatBotTrajectoryStep(
                    step_id=row["step_id"],
                    pre_action_event_id=row["event_id"],
                    action_id=row["action_id"],
                    action_type=row["action_type"],
                    stage=row["stage"],
                    visible_context=row["visible_context"],
                    evidence=tuple(row["evidence"]),
                    internal_signals=_numeric_map(row["internal_signals"]),
                    external_signals=_numeric_map(row["external_signals"]),
                    cost_signals=_numeric_map(row["cost_signals"]),
                    uncertainty_output=_numeric_map(row["uncertainty_output"]),
                    outcome_status=outcome["outcome_status"] if outcome else None,
                    observed_outcome=outcome["observed_outcome"] if outcome else None,
                    realized_cost=float(outcome["realized_cost"]) if outcome else None,
                    terminal=bool(outcome["terminal"]) if outcome else False,
                    agent_backend=row.get("agent_backend"),
                    tool_trace=tuple(row.get("tool_trace", ())),
                    action_text=row.get("action_text"),
                    rationale=row.get("rationale"),
                )
            )
        trajectories.append(
            MatBotTrajectory(
                trajectory_id=trajectory_id,
                run_id=next(iter(run_ids)),
                system=next(iter(systems)),
                capture_mode=next(iter(capture_modes)),
                steps=tuple(steps),
                complete=bool(steps and steps[-1].terminal),
            )
        )
    return trajectories


def trajectory_summary(trajectories: list[MatBotTrajectory]) -> dict[str, Any]:
    return {
        "trajectory_count": len(trajectories),
        "step_count": sum(len(row.steps) for row in trajectories),
        "complete_count": sum(row.complete for row in trajectories),
        "live_runtime_count": sum(row.is_live_runtime for row in trajectories),
        "capture_modes": {
            mode: sum(row.capture_mode == mode for row in trajectories)
            for mode in sorted(CAPTURE_MODES)
        },
    }


def _trajectory_from_json(row: dict[str, Any]) -> MatBotTrajectory:
    capture_mode = row.get("capture_mode")
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"invalid capture mode {capture_mode}")
    step_rows = row.get("steps")
    if not isinstance(step_rows, list):
        raise ValueError("exported trajectory steps must be a list")
    if [step.get("step_id") for step in step_rows] != list(range(len(step_rows))):
        raise ValueError("exported trajectory has non-contiguous step ids")
    steps = tuple(
        MatBotTrajectoryStep(
            step_id=int(step["step_id"]),
            pre_action_event_id=str(step["pre_action_event_id"]),
            action_id=str(step["action_id"]),
            action_type=str(step["action_type"]),
            stage=str(step["stage"]),
            visible_context=str(step["visible_context"]),
            evidence=tuple(step["evidence"]),
            internal_signals=_numeric_map(step["internal_signals"]),
            external_signals=_numeric_map(step["external_signals"]),
            cost_signals=_numeric_map(step["cost_signals"]),
            uncertainty_output=_numeric_map(step["uncertainty_output"]),
            outcome_status=step.get("outcome_status"),
            observed_outcome=step.get("observed_outcome"),
            realized_cost=(
                float(step["realized_cost"])
                if step.get("realized_cost") is not None
                else None
            ),
            terminal=bool(step["terminal"]),
            agent_backend=step.get("agent_backend"),
            tool_trace=tuple(step.get("tool_trace", ())),
            action_text=step.get("action_text"),
            rationale=step.get("rationale"),
        )
        for step in step_rows
    )
    complete = bool(row.get("complete"))
    if complete != bool(steps and steps[-1].terminal):
        raise ValueError("exported trajectory complete flag disagrees with terminal step")
    return MatBotTrajectory(
        trajectory_id=str(row["trajectory_id"]),
        run_id=str(row["run_id"]),
        system=str(row["system"]),
        capture_mode=str(capture_mode),
        steps=steps,
        complete=complete,
    )


def _validate_common(event):
    if event.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {event.get('schema_version')}")
    if event.get("capture_mode") not in CAPTURE_MODES:
        raise ValueError(f"invalid capture mode {event.get('capture_mode')}")


def _validate_pre_action(event):
    if event.get("hidden_outcome_exposed") is not False:
        raise ValueError("pre-action event exposes hidden outcome")
    if "selected_route" in event or "selected_route" in event.get("uncertainty_output", {}):
        raise ValueError("numeric uncertainty events must not contain selected_route")


def _validate_post_outcome(event):
    if not event.get("parent_event_id"):
        raise ValueError("post-outcome event requires parent_event_id")


def _numeric_map(values):
    output = {key: float(value) for key, value in values.items()}
    if any(not isinstance(value, (int, float)) for value in values.values()):
        raise ValueError("signal maps must be numeric")
    return output
