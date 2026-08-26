from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .matbot_trajectory import SCHEMA_VERSION, assemble_matbot_trajectories


REQUIRED_NUMERIC_FIELDS = (
    "p_success",
    "expected_scientific_utility",
    "internal_uncertainty",
    "external_uncertainty",
    "evidence_value",
    "irreversible_risk",
    "estimated_cost",
)


def load_openclaw_matbot_trajectory(
    session_path: str | Path,
    *,
    system: str,
    stage: str,
):
    event = openclaw_session_to_pre_action_event(
        session_path,
        system=system,
        stage=stage,
    )
    return assemble_matbot_trajectories([event])[0]


def openclaw_session_to_pre_action_event(
    session_path: str | Path,
    *,
    system: str,
    stage: str,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(session_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    session = next((row for row in rows if row.get("type") == "session"), None)
    model_change = next(
        (row for row in reversed(rows) if row.get("type") == "model_change"), None
    )
    user_text = ""
    assistant_texts = []
    calls = []
    results = {}
    final_timestamp = None
    for row in rows:
        message = row.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content", [])
        if role == "user" and not user_text:
            user_text = _text_content(content)
        elif role == "assistant":
            final_timestamp = row.get("timestamp", final_timestamp)
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "toolCall":
                    calls.append(
                        {
                            "tool_call_id": str(part.get("id", "")),
                            "tool_name": str(part.get("name", "unknown")),
                            "arguments": part.get("arguments", {}),
                        }
                    )
                elif part.get("type") == "text" and isinstance(part.get("text"), str):
                    assistant_texts.append(part["text"])
        elif role == "toolResult":
            results[str(message.get("toolCallId", ""))] = {
                "tool_name": str(message.get("toolName", "unknown")),
                "status": "error" if message.get("isError") else "success",
                "is_error": bool(message.get("isError")),
            }
    final = _find_numeric_action(assistant_texts)
    tool_trace = []
    for call in calls:
        result = results.get(call["tool_call_id"], {})
        tool_trace.append(
            {
                **call,
                "status": result.get("status", "missing_result"),
                "is_error": bool(result.get("is_error", True)),
            }
        )
    error_count = sum(row["is_error"] for row in tool_trace)
    tool_count = len(tool_trace)
    provider = "unknown"
    model = "unknown"
    if model_change is not None:
        provider = str(model_change.get("provider", provider))
        model = str(model_change.get("modelId", model_change.get("model", model)))
    session_id = str(
        (session or {}).get("id")
        or (session or {}).get("sessionId")
        or Path(session_path).stem
    )
    action = str(final["action"])
    action_digest = hashlib.sha256(action.encode("utf-8")).hexdigest()[:16]
    expected_net_gain = (
        float(final["p_success"]) * float(final["expected_scientific_utility"])
        - float(final["estimated_cost"])
        - float(final["irreversible_risk"]) * (1.0 - float(final["p_success"]))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": "pre_action",
        "event_id": f"openclaw-pre::{session_id}",
        "run_id": session_id,
        "trajectory_id": f"openclaw::{session_id}",
        "step_id": 0,
        "timestamp": final_timestamp or "",
        "capture_mode": "live_runtime",
        "system": system,
        "stage": stage,
        "action_id": f"proposed-experiment::{action_digest}",
        "action_type": "propose_experiment",
        "visible_context": user_text,
        "evidence": [_tool_evidence(row) for row in tool_trace],
        "internal_signals": {
            "reported_internal_uncertainty": float(final["internal_uncertainty"]),
            "tool_hop_fraction": min(1.0, tool_count / 50.0),
        },
        "external_signals": {
            "reported_external_uncertainty": float(final["external_uncertainty"]),
            "tool_error_fraction": error_count / tool_count if tool_count else 0.0,
            "tool_success_fraction": (tool_count - error_count) / tool_count if tool_count else 0.0,
        },
        "cost_signals": {
            "estimated_cost": float(final["estimated_cost"]),
        },
        "uncertainty_output": {
            "p_success": float(final["p_success"]),
            "expected_scientific_utility": float(final["expected_scientific_utility"]),
            "expected_net_scientific_gain": expected_net_gain,
            "reported_internal_uncertainty": float(final["internal_uncertainty"]),
            "reported_external_uncertainty": float(final["external_uncertainty"]),
            "expected_value_of_evidence": float(final["evidence_value"]),
            "irreversible_risk": float(final["irreversible_risk"]),
        },
        "agent_backend": f"{provider}/{model}",
        "tool_trace": tool_trace,
        "action_text": action,
        "rationale": str(final.get("rationale", "")),
        "hidden_outcome_exposed": False,
    }


def _find_numeric_action(texts: list[str]) -> dict[str, Any]:
    for text in reversed(texts):
        for start in reversed([index for index, char in enumerate(text) if char == "{"]):
            try:
                payload, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or "action" not in payload:
                continue
            if not all(field in payload for field in REQUIRED_NUMERIC_FIELDS):
                continue
            for field in REQUIRED_NUMERIC_FIELDS:
                value = payload[field]
                if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"invalid numeric action field {field}")
            return payload
    raise ValueError("OpenClaw session contains no final numeric action JSON")


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def _tool_evidence(row) -> str:
    arguments = row.get("arguments", {})
    target = arguments.get("path") or arguments.get("query") or arguments.get("command") or ""
    target = " ".join(str(target).split())[:240]
    return f"{row['tool_name']}::{row['status']}::{target}"
