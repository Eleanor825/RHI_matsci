from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .features import text_confidence

SCHEMA_VERSION = "materials-action-trajectory-v1"
DEFAULT_TASK_FILES = {
    "matbench_pairwise": "matbench_pairwise_actions.jsonl",
    "discover_unique": "unique_materials_actions.jsonl",
    "extreme_properties": "extreme_properties_actions.jsonl",
}

ORACLE_SIGNAL_KEYS = {
    "preference_strength", "latent_utility", "metric_value", "performance_score",
    "uniqueness_score", "discovery_score", "target_hit_score", "hit_fraction",
    "evidence_support", "evidence_conflict", "verbal_confidence", "tool_agreement",
    "consensus_spread", "ood_score", "reward", "reward_score", "mean_clipped_error",
    "five_hit",
}
ORACLE_CONTEXT_KEYS = {
    "performance_score", "uniqueness_score", "hit_fraction", "reward",
    "reward_score", "mean_clipped_error", "five_hit",
}


def _sanitize_text(task: str, value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if task in {"unique_materials_screening", "materials_pairwise_preference"}:
        text = re.sub(r"log10\(K_VRH\)\s*=\s*[-+]?\d+(?:\.\d+)?", "property estimate", text)
        text = re.sub(r"log10_k_vrh\s*=\s*[-+]?\d+(?:\.\d+)?", "property estimate", text, flags=re.IGNORECASE)
    if task == "extreme_property_discovery":
        text = re.sub(r"(?:hit_fraction|reward|all_hit)\s*=\s*[^,;\s]+", "outcome withheld", text)
    return text


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _trajectory_id(record: dict[str, Any]) -> str:
    task = str(record.get("task", record.get("benchmark", "unknown")))
    group = str(record.get("group_id") or record.get("trace_id") or record["record_id"])
    return f"{task}::{group}"


def _action_index(record: dict[str, Any], fallback: int) -> int:
    value = record.get("event_seq")
    return int(value) if isinstance(value, int) else fallback


def _model_input(record: dict[str, Any]) -> dict[str, Any]:
    """Return fields legal at decision time; outcomes stay in the same row but are isolated."""
    task = str(record.get("task", record.get("benchmark", "")))
    signals = {
        str(key): value
        for key, value in dict(record.get("uncertainty_signals", {})).items()
        if str(key) not in ORACLE_SIGNAL_KEYS and isinstance(value, (int, float))
    }
    context_features = {
        str(key): value
        for key, value in dict(record.get("context_features", {})).items()
        if str(key) not in ORACLE_CONTEXT_KEYS and isinstance(value, (int, float))
    }
    text_probability, text_support, text_conflict = text_confidence(
        f"{record.get('visible_context', '')} {record.get('candidate_action', '')}"
    )
    signals.update({
        "verbal_confidence": text_probability,
        "evidence_support": text_support,
        "evidence_conflict": text_conflict,
    })
    return {
        "visible_context": _sanitize_text(task, record.get("visible_context", "")),
        "candidate_action": _sanitize_text(task, record.get("candidate_action", "")),
        "action_type": record.get("action_type", "recommend"),
        "evidence": [_sanitize_text(task, value) for value in record.get("evidence", [])],
        "context_features": context_features,
        "uncertainty_signals": signals,
        "verbal_confidence": text_probability,
        "cost_level": record.get("cost_level"),
        "reversibility": record.get("reversibility"),
        "risk_level": record.get("risk_level"),
    }


def _outcome(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": int(bool(record.get("outcome_success", record.get("label", False)))),
        "utility": record.get("metric_value", record.get("utility", 0.0)),
        "outcome_success": record.get("outcome_success"),
        "expected_outcome": record.get("expected_outcome"),
        "failure_mode": record.get("failure_mode"),
        "tool_outputs": record.get("tool_outputs", {}),
        "label_source": record.get("label_source"),
        "label_confidence": record.get("label_confidence"),
        "metric_key": record.get("metric_key"),
        "metric_direction": record.get("metric_direction"),
    }


def _pack_action(record: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    trajectory_id = _trajectory_id(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_unit_id": str(record["record_id"]),
        "trajectory_id": trajectory_id,
        "trajectory_type": "benchmark_regime",
        "source_trace_id": record.get("trace_id"),
        "action_index": _action_index(record, fallback_index),
        "action_count": None,
        "trajectory_complete": True,
        "task": record.get("task", record.get("benchmark")),
        "domain": record.get("domain"),
        "model_input": _model_input(record),
        "outcome": _outcome(record),
        "provenance": {
            "record_id": record.get("record_id"),
            "agent_id": record.get("agent_id"),
            "source": record.get("source", {}),
            "notes": record.get("notes"),
            "created_online": bool(record.get("created_online", False)),
        },
        "raw_record": record,
    }


def build_dataset(input_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for path in input_paths:
        for record in _read_jsonl(path):
            records.append(record)
            source_counts[str(record.get("task", "unknown"))] += 1
    records.sort(key=lambda row: (_trajectory_id(row), _action_index(row, 0), str(row["record_id"])))
    packed = [_pack_action(record, index) for index, record in enumerate(records)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in packed:
        grouped[item["trajectory_id"]].append(item)
    trajectories = []
    for trajectory_id, actions in sorted(grouped.items()):
        actions.sort(key=lambda row: (row["action_index"], row["decision_unit_id"]))
        for index, action in enumerate(actions):
            action["action_index"] = index
            action["action_count"] = len(actions)
        first = actions[0]
        trajectories.append({
            "schema_version": SCHEMA_VERSION,
            "trajectory_id": trajectory_id,
            "trajectory_type": "benchmark_regime",
            "trajectory_complete": True,
            "task": first["task"],
            "domain": first["domain"],
            "source_trace_ids": sorted({str(a["source_trace_id"]) for a in actions if a["source_trace_id"]}),
            "action_count": len(actions),
            "actions": actions,
        })
    action_path = out_dir / "actions.jsonl.gz"
    trajectory_path = out_dir / "trajectories.jsonl.gz"
    with gzip.open(action_path, "wt", encoding="utf-8") as handle:
        for item in packed:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with gzip.open(trajectory_path, "wt", encoding="utf-8") as handle:
        for item in trajectories:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": "Complete offline material-science benchmark trajectories with action-level decision units.",
        "decision_unit_definition": "Each action is independently labeled for uncertainty gating; grouped trajectories preserve benchmark-regime order.",
        "input_files": [path.name for path in input_paths],
        "records": len(packed),
        "trajectories": len(trajectories),
        "task_counts": dict(sorted(source_counts.items())),
        "action_count_distribution": dict(sorted(Counter(t["action_count"] for t in trajectories).items())),
        "created_online_count": sum(bool(r.get("created_online", False)) for r in records),
        "label_sources": sorted({str(r.get("label_source", "unknown")) for r in records}),
        "artifacts": {
            "actions": action_path.name,
            "trajectories": trajectory_path.name,
        },
        "leakage_contract": {
            "model_input_excludes": ["outcome", "raw_record.metric_value", "raw_record.tool_outputs", "raw_record.outcome_success"],
            "outcomes_retained_for_training_and_audit": True,
        },
    }
    manifest["sha256"] = {
        name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        for name in [action_path.name, trajectory_path.name]
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package material action records as complete trajectories.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASK_FILES))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.data_dir / DEFAULT_TASK_FILES[task] for task in args.tasks.split(",") if task]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")
    print(json.dumps(build_dataset(paths, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
