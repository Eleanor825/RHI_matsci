from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import binary_metrics, discovery_gain
from .schema import ActionRecord
from .training import TrainedGate, train_gate_with_features


BASE_SIGNALS = (
    "evidence_support",
    "evidence_conflict",
    "source_reliability",
    "tool_agreement",
    "perturbation_stability",
    "verbal_confidence",
    "ood_score",
    "cost",
    "reversibility",
)
ALL_CANDIDATE_SIGNALS = BASE_SIGNALS + (
    "preference_margin",
    "candidate_novelty",
    "candidate_diversity",
    "tail_score",
)
TASK_SIGNAL_PRIORS = {
    "materials_pairwise_preference": ("preference_margin", "tool_agreement", "evidence_support"),
    "unique_materials_screening": ("candidate_novelty", "candidate_diversity", "ood_score"),
    "extreme_property_discovery": ("tail_score", "ood_score", "tool_agreement"),
}
STAGES = ("proposal", "verification", "commit")
ROUTES = ("execute", "verify", "abstain")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stage(row: dict[str, Any]) -> str:
    index = int(row.get("action_index", 0))
    count = max(1, int(row.get("action_count", 1)))
    if index == 0:
        return "proposal"
    if index >= count - 1:
        return "commit"
    return "verification"


def _task(row: dict[str, Any]) -> str:
    task = str(row.get("task") or row.get("benchmark") or "unknown")
    aliases = {
        "matbench_pairwise": "materials_pairwise_preference",
        "discover_unique": "unique_materials_screening",
        "extreme_properties": "extreme_property_discovery",
        "extreme-rlcc": "extreme_property_discovery",
    }
    return aliases.get(task, task)


def _features(row: dict[str, Any], task: str, stage: str) -> dict[str, float]:
    model_input = dict(row.get("model_input", {}))
    values = dict(model_input.get("uncertainty_signals", {}))
    values.update(dict(model_input.get("context_features", {})))
    values["verbal_confidence"] = _scalar(model_input.get("verbal_confidence", values.get("verbal_confidence")))
    values["cost"] = _scalar(values.get("cost"), {"low": 0.2, "medium": 0.5, "high": 0.85}.get(str(model_input.get("cost_level")), 0.5))
    values["reversibility"] = _scalar(values.get("reversibility"), {"low": 0.2, "medium": 0.5, "high": 0.85}.get(str(model_input.get("reversibility")), 0.5))
    values["ood_score"] = _scalar(values.get("ood_score"))
    values["candidate_novelty"] = _scalar(values.get("novelty", values.get("uniqueness", values.get("discovery_score", 0.0))))
    values["candidate_diversity"] = _scalar(values.get("diversity", values.get("composition_diversity", 0.0)))
    values["tail_score"] = _scalar(values.get("tail_probability", values.get("extremeness", values.get("target_hit_score", 0.0))))
    output = {name: _scalar(values.get(name)) for name in BASE_SIGNALS}
    for name in ("preference_margin", "candidate_novelty", "candidate_diversity", "tail_score"):
        output[name] = _scalar(values.get(name))
    for candidate_task in sorted(TASK_SIGNAL_PRIORS):
        output[f"task__{candidate_task}"] = float(task == candidate_task)
    for candidate_stage in STAGES:
        output[f"stage__{candidate_stage}"] = float(stage == candidate_stage)
    return output


def load_action_records(path: str | Path) -> list[ActionRecord]:
    records = []
    for row in _read_jsonl(Path(path)):
        model_input = dict(row.get("model_input", {}))
        outcome = dict(row.get("outcome", {}))
        task = _task(row)
        stage = _stage(row)
        features = _features(row, task, stage)
        metadata = {
            "trajectory_id": str(row.get("trajectory_id", "unknown")),
            "task": task,
            "stage": stage,
            "action_index": int(row.get("action_index", 0)),
            "action_count": int(row.get("action_count", 1)),
        }
        records.append(ActionRecord(
            record_id=str(row.get("decision_unit_id", row.get("record_id"))),
            benchmark=task,
            split="unspecified",
            visible_context=str(model_input.get("visible_context", ""))[:800],
            candidate_action=str(model_input.get("candidate_action", ""))[:800],
            action_type=str(model_input.get("action_type", "recommend")),
            evidence=[str(item)[:300] for item in model_input.get("evidence", [])],
            features=features,
            label=int(bool(outcome.get("label", 0))),
            utility=_scalar(outcome.get("utility", 0.0)),
            metadata=metadata,
        ))
    if not records:
        raise ValueError(f"no action records found in {path}")
    return records


def _split_trajectories(records: list[ActionRecord], seed: int) -> dict[str, list[ActionRecord]]:
    groups: dict[str, list[ActionRecord]] = defaultdict(list)
    for record in records:
        groups[str(record.metadata["trajectory_id"])].append(record)
    ids = list(groups)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    boundaries = (max(1, int(n * 0.60)), max(2, int(n * 0.75)), max(3, int(n * 0.85)))
    if boundaries[-1] >= n:
        boundaries = (max(1, n - 3), max(2, n - 2), max(3, n - 1))
    names = ("train", "feedback", "acceptance", "test")
    output = {name: [] for name in names}
    for index, trajectory_id in enumerate(ids):
        bucket = 0 if index < boundaries[0] else 1 if index < boundaries[1] else 2 if index < boundaries[2] else 3
        output[names[bucket]].extend(groups[trajectory_id])
    if any(not output[name] for name in names):
        raise ValueError(f"trajectory split is too small: { {name: len(rows) for name, rows in output.items()} }")
    return output


def feature_policy(task: str, stage: str, active_signals: set[str]) -> list[str]:
    selected = set(active_signals)
    selected.update(TASK_SIGNAL_PRIORS.get(task, ()))
    if stage == "proposal":
        selected -= {"tool_agreement"}
    if stage == "verification":
        selected.update({"tool_agreement", "source_reliability", "evidence_conflict"})
    if stage == "commit":
        selected.update({"cost", "reversibility", "ood_score", "source_reliability"})
    selected &= set(ALL_CANDIDATE_SIGNALS)
    names = sorted(selected)
    for signal in tuple(names):
        names.append(f"{signal}__task__{task}")
        names.append(f"{signal}__stage__{stage}")
    return sorted(set(names))


def add_policy_features(records: list[ActionRecord], active_signals: set[str]) -> list[ActionRecord]:
    output = []
    for record in records:
        task = str(record.metadata["task"])
        stage = str(record.metadata["stage"])
        features = dict(record.features)
        selected = feature_policy(task, stage, active_signals)
        for name in active_signals | set(BASE_SIGNALS) | set(TASK_SIGNAL_PRIORS.get(task, ())):
            value = _scalar(features.get(name))
            features[f"{name}__task__{task}"] = value
            features[f"{name}__stage__{stage}"] = value
        features["stage_progress"] = _scalar(record.metadata["action_index"]) / max(1.0, _scalar(record.metadata["action_count"]) - 1.0)
        features["is_commit"] = float(stage == "commit")
        features["is_verification"] = float(stage == "verification")
        output.append(ActionRecord(record.record_id, record.benchmark, record.split, record.visible_context, record.candidate_action, record.action_type, record.evidence, features, record.label, record.utility, record.metadata))
    return output


def _feature_names(records: list[ActionRecord], active_signals: set[str]) -> list[str]:
    names = set()
    for record in records:
        names.update(feature_policy(str(record.metadata["task"]), str(record.metadata["stage"]), active_signals))
        names.update({"stage_progress", "is_commit", "is_verification"})
    return sorted(names)


def _predict(records: list[ActionRecord], gate: TrainedGate) -> list[float]:
    return gate.predict_proba(records)


def _report(records: list[ActionRecord], gate: TrainedGate, budget_fraction: float) -> dict[str, Any]:
    probabilities = _predict(records, gate)
    labels = [record.label for record in records]
    budget = max(1, int(len(records) * budget_fraction))
    metrics = binary_metrics(labels, probabilities, gate.threshold)
    gain = discovery_gain(labels, [record.utility for record in records], probabilities, budget)
    return {"metrics": metrics, "discovery_gain": gain, "threshold": gate.threshold, "n": len(records)}


def _stable_failure_patterns(records: list[ActionRecord], probabilities: list[float], threshold: float, min_group: int = 20) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(str(record.metadata["task"]), str(record.metadata["stage"]))].append(index)
    residuals = [abs(probabilities[i] - records[i].label) for i in range(len(records))]
    global_error = sum(residuals) / max(1, len(residuals))
    patterns = []
    for (task, stage), indices in groups.items():
        if len(indices) < min_group:
            continue
        group_error = sum(residuals[i] for i in indices) / len(indices)
        if group_error < global_error + 0.02:
            continue
        effects = {}
        for signal in ALL_CANDIDATE_SIGNALS:
            values = [records[i].features.get(signal, 0.0) for i in indices]
            if values:
                effects[signal] = abs(sum(records[i].features.get(signal, 0.0) * residuals[i] for i in indices) / len(indices) - sum(values) / len(values) * group_error)
        patterns.append({"task": task, "stage": stage, "n": len(indices), "mean_abs_error": group_error, "global_mean_abs_error": global_error, "threshold": threshold, "effects": effects})
    return patterns


def _score(report: dict[str, Any]) -> float:
    metrics = report["metrics"]
    gain = report["discovery_gain"]
    return 0.35 * gain.get("utility_efficiency", 0.0) + 0.25 * gain.get("hit_efficiency", 0.0) + 0.20 * (1.0 - metrics.get("selective_risk", 1.0)) + 0.20 * (1.0 - metrics.get("brier", 1.0))


def run_conditional_action_rhi(records: list[ActionRecord], *, iterations: int = 3, seed: int = 1729, alpha: float = 0.10, budget_fraction: float = 0.10, epochs: int = 120) -> dict[str, Any]:
    splits = _split_trajectories(records, seed)
    active_signals = {"evidence_support", "verbal_confidence", "cost", "reversibility"}
    versions = []
    patterns_by_round = []
    current = None
    for round_index in range(iterations + 1):
        prepared = {name: add_policy_features(rows, active_signals) for name, rows in splits.items()}
        names = _feature_names(prepared["train"] + prepared["feedback"], active_signals)
        gate = train_gate_with_features(prepared["train"], prepared["feedback"], feature_names=names, alpha=alpha, epochs=epochs, learning_rate=0.08, l2=0.01, min_coverage=budget_fraction)
        acceptance = _report(prepared["acceptance"], gate, budget_fraction)
        test = _report(prepared["test"], gate, budget_fraction)
        feedback_probabilities = _predict(prepared["feedback"], gate)
        patterns = _stable_failure_patterns(prepared["feedback"], feedback_probabilities, gate.threshold) if round_index < iterations else []
        versions.append({"round": round_index, "active_signals": sorted(active_signals), "feature_count": len(names), "acceptance": acceptance, "test": test, "score": _score(test)})
        patterns_by_round.append(patterns)
        if not patterns:
            continue
        candidate = set(active_signals)
        for pattern in patterns:
            ranked = sorted(pattern["effects"].items(), key=lambda item: item[1], reverse=True)
            candidate.update(name for name, effect in ranked[:2] if effect >= 0.03)
        candidate_prepared = {name: add_policy_features(rows, candidate) for name, rows in splits.items()}
        candidate_names = _feature_names(candidate_prepared["train"] + candidate_prepared["feedback"], candidate)
        candidate_gate = train_gate_with_features(candidate_prepared["train"], candidate_prepared["feedback"], feature_names=candidate_names, alpha=alpha, epochs=epochs, learning_rate=0.08, l2=0.01, min_coverage=budget_fraction)
        candidate_acceptance = _report(candidate_prepared["acceptance"], candidate_gate, budget_fraction)
        if _score(candidate_acceptance) >= _score(acceptance) and candidate_acceptance["metrics"]["selective_risk"] <= alpha + 0.05:
            active_signals = candidate
        else:
            break
    return {"method": "Conditional Action-Worthiness RHI", "single_agent": True, "split_sizes": {name: len(rows) for name, rows in splits.items()}, "iterations": iterations, "budget_fraction": budget_fraction, "alpha": alpha, "versions": versions, "stable_failure_patterns": patterns_by_round, "final_signals": sorted(active_signals)}


def run_experiment(input_path: str | Path, out_dir: str | Path, *, seed: int = 1729, iterations: int = 3, epochs: int = 120) -> dict[str, Any]:
    records = load_action_records(input_path)
    result = run_conditional_action_rhi(records, seed=seed, iterations=iterations, epochs=epochs)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Conditional Action-Worthiness RHI", "", f"- Single agent action units: {result['single_agent']}", f"- Split sizes: {result['split_sizes']}", f"- Final active signals: {', '.join(result['final_signals'])}", "- Test is evaluated only after acceptance; benchmark labels are offline proxies.", "", "| Round | Features | Acceptance score | Test score | Test risk | Test coverage |", "|---:|---:|---:|---:|---:|---:|"]
    for version in result["versions"]:
        metrics = version["test"]["metrics"]
        lines.append(f"| {version['round']} | {version['feature_count']} | {_score(version['acceptance']):.4f} | {version['score']:.4f} | {metrics['selective_risk']:.4f} | {metrics['coverage']:.4f} |")
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-agent conditional action-worthiness RHI.")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    result = run_experiment(args.actions, args.out_dir, seed=args.seed, iterations=args.iterations, epochs=args.epochs)
    print(json.dumps({"split_sizes": result["split_sizes"], "final_signals": result["final_signals"], "versions": result["versions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
