from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .direct_judge import (
    DEFAULT_USER_AGENT,
    DirectJudgeError,
    _default_request,
    _parse_json_object,
    _response_content,
)


PROMPT_VERSION = "scientific-tool-policy-v1"
TERMINAL_ACTIONS = {"execute_candidate", "abstain"}
STAGE_ACTIONS = {
    "acquire_screening": "screening",
    "acquire_verification": "verification",
}


@dataclass(frozen=True)
class ToolPolicyDecision:
    action_id: str
    confidence: float
    rationale: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class ToolPolicy(Protocol):
    model: str

    def decide(
        self,
        *,
        cache_key: str,
        system_prompt: str,
        user_prompt: str,
        allowed_actions: Sequence[str],
    ) -> ToolPolicyDecision: ...


class LLMToolPolicyClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        reasoning_effort: str = "medium",
        cache_path: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        request_fn: Callable[[Request, float], bytes] | None = None,
    ) -> None:
        if not model or not api_key or not base_url:
            raise ValueError("model, api_key, and base_url are required")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.cache_path = Path(cache_path) if cache_path else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_fn = request_fn or _default_request
        self._cache = self._load_cache()
        self._cache_lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        cache_path: str | Path | None = None,
        reasoning_effort: str = "medium",
    ) -> "LLMToolPolicyClient":
        return cls(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            reasoning_effort=reasoning_effort,
            cache_path=cache_path,
        )

    def decide(
        self,
        *,
        cache_key: str,
        system_prompt: str,
        user_prompt: str,
        allowed_actions: Sequence[str],
    ) -> ToolPolicyDecision:
        cached = self._cache.get(cache_key)
        if cached is not None:
            return ToolPolicyDecision(**cached)
        prediction = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(system_prompt, user_prompt, allowed_actions)
                prediction = _parse_decision(response, allowed_actions)
                break
            except (DirectJudgeError, HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
        if prediction is None:
            raise DirectJudgeError(
                f"tool-policy response remained invalid after retries: {last_error}"
            ) from last_error
        with self._cache_lock:
            self._cache[cache_key] = prediction.to_json()
            self._save_cache()
        return prediction

    def _request(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed_actions: Sequence[str],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
            "stream": False,
            "max_output_tokens": 384,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "scientific_tool_policy_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "string", "enum": list(allowed_actions)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string"},
                        },
                        "required": ["action_id", "confidence", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
        }
        request = Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
                "OpenAI-Beta": "responses=v1",
            },
            method="POST",
        )
        decoded = json.loads(self.request_fn(request, self.timeout).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise DirectJudgeError("tool-policy API response is not a JSON object")
        return decoded

    def _load_cache(self) -> dict[str, dict[str, object]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        contract = (
            payload.get("model"),
            payload.get("prompt_version"),
            payload.get("reasoning_effort"),
        )
        if contract != (self.model, PROMPT_VERSION, self.reasoning_effort):
            return {}
        return dict(payload.get("decisions", {}))

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "reasoning_effort": self.reasoning_effort,
            "decisions": dict(sorted(self._cache.items())),
        }
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)


def run_model_tool_trajectory(
    source: dict[str, object],
    policy: ToolPolicy,
    *,
    max_steps: int = 3,
    evidence_cost_weight: float = 0.15,
) -> dict[str, object]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    stages = {str(row["stage"]): row for row in source["stages"]}
    if "base" not in stages:
        raise ValueError("source trajectory lacks base stage")
    current_stage = "base"
    acquired = {"base"}
    history: list[dict[str, object]] = []
    steps = []
    evidence_cost = 0.0
    terminal_action = "abstain"
    for step_index in range(max_steps):
        allowed = _allowed_actions(acquired, stages)
        visible = stages[current_stage]["visible_observation"]
        system_prompt = _system_prompt()
        user_prompt = _user_prompt(
            source=source,
            current_stage=current_stage,
            visible=visible,
            history=history,
            allowed_actions=allowed,
        )
        cache_key = f"{source['record_id']}::{current_stage}::{step_index}"
        decision = policy.decide(
            cache_key=cache_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_actions=allowed,
        )
        steps.append(
            {
                "step": step_index,
                "stage": current_stage,
                "agent_backend": policy.model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "available_actions": list(allowed),
                "agent_output": decision.to_json(),
                "visible_observation": visible,
                "hidden_outcome_exposed": False,
            }
        )
        history.append({"step": step_index, "action_id": decision.action_id})
        if decision.action_id in TERMINAL_ACTIONS:
            terminal_action = decision.action_id
            break
        next_stage = STAGE_ACTIONS[decision.action_id]
        acquired.add(next_stage)
        current_stage = next_stage
        evidence_cost += float(stages[next_stage]["tool_cost"])
    else:
        terminal_action = "abstain"

    hidden = dict(source["hidden_outcome_for_evaluation_only"])
    executed = terminal_action == "execute_candidate"
    base_gain = float(hidden["base_gain"])
    realized_gain = (
        base_gain - evidence_cost_weight * evidence_cost
        if executed
        else -evidence_cost_weight * evidence_cost
    )
    return {
        "schema_version": "model-tool-trajectory-v1",
        "trajectory_id": f"model-tool::{policy.model}::{source['record_id']}",
        "source_record_id": source["record_id"],
        "benchmark": source["benchmark"],
        "split": source["split"],
        "agent_backend": policy.model,
        "prompt_version": PROMPT_VERSION,
        "capture_mode": "live_model_controlled_tool_replay",
        "online_model": True,
        "matbot_runtime": False,
        "steps": steps,
        "terminal_action": terminal_action,
        "executed_candidate": executed,
        "evidence_cost": evidence_cost,
        "hidden_outcome_for_evaluation_only": {
            **hidden,
            "hidden_from_prompts": True,
            "trajectory_strict_net_gain": realized_gain,
        },
        "credit_assignment": {
            "terminal_execute_has_direct_outcome": executed,
            "evidence_actions_receive_trajectory_return_only": True,
            "causal_action_value_claimed": False,
        },
    }


def _allowed_actions(
    acquired: set[str],
    stages: dict[str, dict[str, object]],
) -> tuple[str, ...]:
    actions = ["execute_candidate"]
    if "screening" in stages and "screening" not in acquired:
        actions.append("acquire_screening")
    if "verification" in stages and "verification" not in acquired:
        actions.append("acquire_verification")
    actions.append("abstain")
    return tuple(actions)


def _system_prompt() -> str:
    return (
        "You are a materials-science discovery agent controlling real model calls "
        "over a controlled scientific tool replay. Choose exactly one allowed action "
        "using only visible pre-action information. Hidden outcomes, benchmark labels, "
        "and future tool observations are unavailable. Return only the required JSON."
    )


def _user_prompt(
    *,
    source: dict[str, object],
    current_stage: str,
    visible: dict[str, object],
    history: Sequence[dict[str, object]],
    allowed_actions: Sequence[str],
) -> str:
    record = visible["record"]
    evidence = "\n".join(f"- {item}" for item in record.get("evidence", [])) or "- none"
    action_descriptions = {
        "execute_candidate": str(record["candidate_action"]),
        "acquire_screening": "Acquire the low-cost screening-tool observation.",
        "acquire_verification": "Acquire the higher-fidelity verification-tool observation.",
        "abstain": "Stop without executing the candidate.",
    }
    menu = "\n".join(
        f"- {action_id}: {action_descriptions[action_id]}"
        for action_id in allowed_actions
    )
    prior = "\n".join(
        f"- step {row['step']}: {row['action_id']}" for row in history
    ) or "- none"
    return f"""Task: {source['benchmark']}
Current evidence stage: {current_stage}

Visible scientific context:
<context>
{record['visible_context']}
</context>

Visible evidence:
<evidence>
{evidence}
</evidence>

Prior actions:
{prior}

Allowed actions:
{menu}

Choose one action. Confidence is confidence in the action choice, not a hidden-outcome probability."""


def _parse_decision(
    response: dict[str, Any],
    allowed_actions: Sequence[str],
) -> ToolPolicyDecision:
    payload = _parse_json_object(_response_content(response))
    action_id = str(payload.get("action_id", ""))
    if action_id not in allowed_actions:
        raise DirectJudgeError(f"tool-policy action is not allowed: {action_id}")
    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise DirectJudgeError("tool-policy confidence is invalid")
    return ToolPolicyDecision(
        action_id=action_id,
        confidence=float(confidence),
        rationale=str(payload.get("rationale", "")),
    )
