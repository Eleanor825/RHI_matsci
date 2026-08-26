from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .historical import MAIN_MATERIAL_TASKS, load_historical_task_records
from .schema import ActionRecord
from .signal_contract import EXTERNAL_UNCERTAINTY_SIGNALS, INTERNAL_UNCERTAINTY_SIGNALS

TASK_RUBRICS = {
    "matbench_pairwise": "Choose a high-bulk-modulus materials follow-up while accounting for noisy surrogate evidence.",
    "discover_unique": "Find a material that jointly offers high performance and chemical/structural uniqueness.",
    "extreme_properties": "Advance a candidate only when it is worth testing against the extreme-property target constraints.",
}


def _task_name(task: str) -> str:
    return {
        "matbench_pairwise": "materials_pairwise_preference",
        "discover_unique": "unique_materials_screening",
        "extreme_properties": "extreme_property_discovery",
    }.get(task, task)


def _system_prompt(task: str) -> str:
    return f"""You are a materials-science discovery agent.\n\nGoal: {TASK_RUBRICS[task]}\n\nAt each step, choose exactly one action from the available action menu. Do not assume access to hidden benchmark outcomes, true material properties, or post-execution rewards. Use only the visible context and evidence. Prefer verification before an expensive or irreversible action when uncertainty is reducible. Return JSON with action, confidence, and rationale."""


def _action_menu(record: ActionRecord) -> list[dict[str, Any]]:
    return [
        {"action_id": "execute_candidate", "type": record.action_type, "description": record.candidate_action},
        {"action_id": "retrieve_more", "type": "retrieve_more", "description": "Retrieve analogous materials and additional pre-execution evidence."},
        {"action_id": "simulate", "type": "simulate", "description": "Run a low-fidelity simulation or ensemble estimate before committing."},
        {"action_id": "ask_expert", "type": "ask_expert", "description": "Ask a domain expert to review the candidate and evidence."},
        {"action_id": "abstain", "type": "abstain", "description": "Do not execute this candidate in the current state."},
    ]


def _prompt(record: ActionRecord, step: int, prior_actions: list[dict[str, Any]], extra_evidence: list[str]) -> str:
    evidence = list(record.evidence) + list(extra_evidence)
    menu = "\n".join(f"- {item['action_id']}: {item['description']}" for item in _action_menu(record))
    history = "\n".join(f"- step {item['step']}: {item['action_id']}" for item in prior_actions) or "- none"
    evidence_text = "\n".join(f"- {item}" for item in evidence) or "- none"
    return f"""Scientific task: {_task_name(record.benchmark)}\nStep: {step}\n\nVisible scientific context:\n<context>\n{record.visible_context}\n</context>\n\nEvidence available before execution:\n<evidence>\n{evidence_text}\n</evidence>\n\nPrevious actions:\n{history}\n\nAvailable actions:\n{menu}\n\nReturn exactly one JSON object: {{\"action_id\": string, \"confidence\": number between 0 and 1, \"rationale\": string}}."""


def _reliability(record: ActionRecord) -> float:
    f = record.features
    score = (
        0.30 * f.get("verbal_confidence", 0.5)
        + 0.25 * f.get("evidence_support", 0.5)
        + 0.20 * f.get("source_reliability", 0.5)
        + 0.15 * f.get("perturbation_stability", 0.5)
        + 0.10 * f.get("tool_agreement", 0.5)
        - 0.20 * f.get("evidence_conflict", 0.0)
        - 0.15 * f.get("ood_score", 0.0)
        - 0.10 * f.get("model_disagreement", 0.0)
    )
    return max(0.0, min(1.0, score))


def _deterministic_action(record: ActionRecord, step: int, prior_actions: list[dict[str, Any]]) -> tuple[str, float, str]:
    reliability = _reliability(record)
    f = record.features
    if step == 0:
        if f.get("ood_score", 0.0) >= 0.70 or f.get("evidence_conflict", 0.0) >= 0.45:
            return "retrieve_more", reliability, "Visible evidence is conflicting or out of distribution."
        if f.get("cost", 0.0) >= 0.70 or f.get("reversibility", 1.0) <= 0.30 or f.get("model_disagreement", 0.0) >= 0.65:
            return "simulate", reliability, "The action is costly, hard to reverse, or model disagreement is high."
        if reliability >= 0.58:
            return "execute_candidate", reliability, "The available evidence supports executing the candidate."
        return "retrieve_more", reliability, "Confidence is insufficient for immediate execution."
    if reliability >= 0.52:
        return "execute_candidate", reliability, "Verification did not remove the need to proceed, but the action is sufficiently supported."
    if f.get("cost", 0.0) >= 0.70:
        return "ask_expert", reliability, "The candidate remains too uncertain for a costly action."
    return "abstain", reliability, "The candidate remains below the execution reliability threshold."


def _sample_records(data_dir: str | Path, task: str, n: int, seed: int) -> list[ActionRecord]:
    records = load_historical_task_records(data_dir, task)
    if len(records) < n:
        raise ValueError(f"{task} has only {len(records)} records; cannot sample {n}")
    rng = random.Random(seed)
    selected = rng.sample(records, n)
    selected.sort(key=lambda record: record.record_id)
    return selected


def generate_prompt_trajectories(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    n_per_task: int = 300,
    seed: int = 1729,
    max_steps: int = 3,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trajectories: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for task_index, task in enumerate(MAIN_MATERIAL_TASKS):
        records = _sample_records(data_dir, task, n_per_task, seed + task_index)
        task_counts[task] = len(records)
        for record_index, record in enumerate(records):
            trajectory_id = f"prompt-traj::{task}::{seed}::{record_index:04d}"
            system_prompt = _system_prompt(task)
            prior_actions: list[dict[str, Any]] = []
            extra_evidence: list[str] = []
            steps: list[dict[str, Any]] = []
            final_action = "abstain"
            for step_index in range(max_steps):
                user_prompt = _prompt(record, step_index, prior_actions, extra_evidence)
                action_id, confidence, rationale = _deterministic_action(record, step_index, prior_actions)
                step = {
                    "step": step_index,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "agent_backend": "deterministic_prompt_policy",
                    "agent_output": {
                        "action_id": action_id,
                        "confidence": confidence,
                        "rationale": rationale,
                    },
                    "available_actions": _action_menu(record),
                    "internal_signals": {
                        key: record.features[key]
                        for key in INTERNAL_UNCERTAINTY_SIGNALS
                        if key in record.features
                    },
                    "external_signals": {
                        key: record.features[key]
                        for key in EXTERNAL_UNCERTAINTY_SIGNALS
                        if key in record.features
                    },
                }
                steps.append(step)
                prior_actions.append({"step": step_index, "action_id": action_id})
                final_action = action_id
                if action_id == "execute_candidate":
                    break
                if action_id == "retrieve_more":
                    extra_evidence.append("Retrieved analog evidence; exact hidden target remains unavailable.")
                elif action_id == "simulate":
                    extra_evidence.append("Simulation produced a noisy consistency check; exact hidden target remains unavailable.")
                else:
                    break
            executed = final_action == "execute_candidate"
            cost = sum(float(record.features.get("cost", 0.5)) for step in steps if step["agent_output"]["action_id"] in {"execute_candidate", "simulate", "ask_expert"})
            trajectory = {
                "schema_version": "prompt-generated-materials-trajectory-v1",
                "trajectory_id": trajectory_id,
                "task": task,
                "benchmark": record.benchmark,
                "source_record_id": record.record_id,
                "regime_id": record.metadata.get("group_id", record.benchmark),
                "agent_backend": "deterministic_prompt_policy",
                "prompt_generated": True,
                "llm_generated": False,
                "max_steps": max_steps,
                "steps": steps,
                "terminal_action": final_action,
                "executed_candidate": executed,
                "outcome": {
                    "label": record.label if executed else 0,
                    "utility": record.utility if executed and record.label else 0.0,
                    "source_label": record.metadata.get("label_source", "historical_benchmark_proxy"),
                    "hidden_from_prompt": True,
                },
                "cost": cost,
                "net_utility_proxy": (record.utility if executed and record.label else 0.0) - 0.15 * min(1.0, cost),
                "source_action_record": record.to_json(),
            }
            trajectories.append(trajectory)
    trajectory_path = out_path / "trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "prompt-generated-materials-trajectory-v1",
        "description": "Prompt-conditioned, replayable multi-step trajectories built from the three main materials tasks.",
        "agent_backend": "deterministic_prompt_policy",
        "prompt_generated": True,
        "llm_generated": False,
        "n_per_task": n_per_task,
        "max_steps": max_steps,
        "seed": seed,
        "trajectory_count": len(trajectories),
        "task_counts": task_counts,
        "leakage_contract": {
            "prompt_excludes": ["label", "utility", "outcome_success", "metric_value", "raw oracle fields"],
            "outcomes_retained_for_evaluation": True,
        },
        "artifacts": {"trajectories": trajectory_path.name},
    }
    evaluation = summarize_prompt_trajectories(trajectories)
    manifest["evaluation"] = evaluation
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out_path / "evaluation_summary.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n")
    (out_path / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


def summarize_prompt_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        executed = [row for row in rows if row["executed_candidate"]]
        successes = sum(int(row["outcome"]["label"]) for row in executed)
        return {
            "n_trajectories": len(rows),
            "execute_rate": len(executed) / len(rows) if rows else 0.0,
            "executed_success_rate": successes / len(executed) if executed else 0.0,
            "executed_selective_risk": 1.0 - successes / len(executed) if executed else 0.0,
            "mean_realized_utility": statistics.mean(row["outcome"]["utility"] for row in executed) if executed else 0.0,
            "mean_net_utility_proxy": statistics.mean(row["net_utility_proxy"] for row in rows) if rows else 0.0,
            "mean_steps": statistics.mean(len(row["steps"]) for row in rows) if rows else 0.0,
            "terminal_actions": {
                action: sum(row["terminal_action"] == action for row in rows)
                for action in sorted({row["terminal_action"] for row in rows})
            },
        }

    return {
        "overall": summarize(trajectories),
        "by_task": {
            task: summarize([row for row in trajectories if row["task"] == task])
            for task in sorted({row["task"] for row in trajectories})
        },
    }


def _readme(manifest: dict[str, Any]) -> str:
    return f"""# Prompt-Conditioned Materials Trajectories v1\n\nThis run contains {manifest['trajectory_count']} prompt-conditioned trajectories: 300 instances from each of the three main materials tasks. Each trajectory has at most {manifest['max_steps']} steps and records the full system prompt, user prompt, available action menu, agent output, internal/external signals, route, and hidden benchmark outcome.\n\nThis first run uses a deterministic prompt policy (`agent_backend=deterministic_prompt_policy`) so the trajectory construction is reproducible without an API key. `prompt_generated=true` means the action is selected from a prompt-defined action menu; `llm_generated=false` is explicit. The same runner can later replace the policy with an LLM backend without changing the trajectory schema.\n\nThe prompt never includes label, utility, post-action reward, or raw oracle fields. Outcomes are retained only for evaluation.\n\nTask counts: `{json.dumps(manifest['task_counts'], sort_keys=True)}`.\n\nThe deterministic policy is a construction baseline, not the proposed harness. Replay the same prompts with no-harness, static-gate, and Sci-VoI-RHI policies before drawing method conclusions.\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt-conditioned materials trajectories.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-per-task", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(generate_prompt_trajectories(args.data_dir, args.out_dir, n_per_task=args.n_per_task, seed=args.seed, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    main()
