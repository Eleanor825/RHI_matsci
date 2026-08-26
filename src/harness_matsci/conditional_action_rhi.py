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
from .calibration import threshold_for_grouped_selective_risk
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
    "agent_prior_margin",
    "agent_prior_uncertainty",
    "surrogate_margin",
    "surrogate_uncertainty",
)
TASK_SIGNAL_PRIORS = {
    "materials_pairwise_preference": ("agent_prior_margin", "agent_prior_uncertainty", "evidence_support", "perturbation_stability"),
    "unique_materials_screening": ("evidence_support", "ood_score", "perturbation_stability", "source_reliability"),
    "extreme_property_discovery": ("surrogate_margin", "surrogate_uncertainty", "ood_score", "evidence_support"),
}
STAGES = ("proposal", "verification", "commit")
ROUTES = ("execute", "verify", "abstain")


class TaskConditionalGate:
    """A collection of task-specific ranking heads with a global fallback."""

    def __init__(self, fallback: TrainedGate, task_gates: dict[str, TrainedGate]):
        self.fallback = fallback
        self.task_gates = task_gates
        self.threshold = fallback.threshold
        self.calibration = {"mode": "task_conditional", "n_task_heads": len(task_gates)}

    def predict_proba(self, records: list[ActionRecord]) -> list[float]:
        probabilities = [0.0] * len(records)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[str(record.metadata.get("task", record.benchmark))].append(index)
        for task, indices in grouped.items():
            gate = self.task_gates.get(task, self.fallback)
            task_records = [records[index] for index in indices]
            task_probabilities = gate.predict_proba(task_records)
            for index, probability in zip(indices, task_probabilities):
                probabilities[index] = probability
        return probabilities


def _top_budget_ids(records: list[ActionRecord], probabilities: list[float], budget_fraction: float) -> set[str]:
    budget = max(1, min(len(records), math.ceil(len(records) * budget_fraction))) if records else 0
    selected = sorted(range(len(records)), key=lambda index: (-probabilities[index], str(records[index].record_id)))[:budget]
    return {str(records[index].record_id) for index in selected}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
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


def load_external_split_records(root: str | Path, *, fold: int = 0) -> dict[str, list[ActionRecord]]:
    """Load the repository's richer external train/feedback/acceptance/test fold."""
    root = Path(root)
    fold_dir = root / "folds" / f"fold_{fold}"
    if not fold_dir.exists():
        fold_dir = root / f"fold_{fold}"
    root_name = root.name.lower()
    preferred_prefix = {
        "external_h14": "external_extreme_phonons",
        "external_h15_unique": "external_unique_log_kvrh",
        "external_h16_pairwise": "external_pairwise_log_kvrh",
        "external_h17_pairwise": "external_pairwise_mixed_log_kvrh",
    }.get(root_name)
    if preferred_prefix is None:
        candidates = sorted(fold_dir.glob("*.train.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no train split in {fold_dir}")
        preferred_prefix = candidates[0].name.removesuffix(".train.jsonl")
    files = {
        split: fold_dir / f"{preferred_prefix}.{split}.jsonl"
        for split in ("train", "feedback", "acceptance", "test")
    }
    files = {split: path if path.exists() else None for split, path in files.items()}
    if any(path is None for path in files.values()):
        raise FileNotFoundError(f"missing external split in {fold_dir}: {files}")
    aliases = {
        "external_pairwise_log_kvrh": "materials_pairwise_preference",
        "external_pairwise_mixed_log_kvrh": "materials_pairwise_preference",
        "external_pairwise_dielectric": "materials_pairwise_preference",
        "external_pairwise_mixed_dielectric": "materials_pairwise_preference",
        "external_unique_log_kvrh": "unique_materials_screening",
        "external_unique_jdft2d": "unique_materials_screening",
        "external_extreme_phonons": "extreme_property_discovery",
        "external_extreme_expt_gap": "extreme_property_discovery",
        "external_extreme_superconductivity": "extreme_property_discovery",
    }
    output: dict[str, list[ActionRecord]] = {}
    for split, path in files.items():
        rows = _read_jsonl(path)
        converted = []
        for index, row in enumerate(rows):
            benchmark = str(row.get("benchmark", path.stem))
            task = aliases.get(benchmark, _task(row))
            metadata = dict(row.get("metadata", {}))
            group_id = str(metadata.get("group_id", row.get("record_id", index)))
            metadata.update({"task": task, "stage": str(metadata.get("stage", "proposal")), "trajectory_id": f"{task}::{group_id}", "action_index": 0, "action_count": 1})
            features = {str(key): _scalar(value) for key, value in dict(row.get("features", {})).items() if isinstance(value, (int, float))}
            converted.append(ActionRecord(
                record_id=str(row.get("record_id", f"{benchmark}-{split}-{index}")),
                benchmark=task,
                split=split,
                visible_context=str(row.get("visible_context", ""))[:1200],
                candidate_action=str(row.get("candidate_action", ""))[:1200],
                action_type=str(row.get("action_type", "choose_candidate")),
                evidence=[str(item)[:500] for item in row.get("evidence", [])],
                features=features,
                label=int(row.get("label", 0)),
                utility=_scalar(row.get("utility", 0.0)),
                metadata=metadata,
            ))
        output[split] = converted
    return output


def run_external_three_task_experiment(roots: dict[str, str | Path], *, fold: int = 0, iterations: int = 3, alpha: float = 0.10, budget_fraction: float = 0.10, epochs: int = 100) -> dict[str, Any]:
    """Run conditional RHI on pre-split external pairwise/unique/extreme tasks."""
    splits = {name: [] for name in ("train", "feedback", "acceptance", "test")}
    task_counts = {}
    for task_name, root in roots.items():
        loaded = load_external_split_records(root, fold=fold)
        task_counts[task_name] = {split: len(rows) for split, rows in loaded.items()}
        for split in splits:
            splits[split].extend(loaded[split])
    active_signals = {"verbal_confidence", "cost", "reversibility"}
    active_policy = "global_score"
    versions = []
    mutations = []
    accepted_rounds = [0]
    for round_index in range(iterations + 1):
        prepared = {name: add_policy_features(rows, active_signals) for name, rows in splits.items()}
        names = _feature_names(prepared["train"] + prepared["feedback"], active_signals)
        if active_policy == "task_conditional_score_heads":
            gate = _train_task_conditional_gate(prepared["train"], prepared["feedback"], active_signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
        else:
            gate = _train_external_gate(prepared["train"], prepared["feedback"], names, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
        acceptance = _report(prepared["acceptance"], gate, budget_fraction)
        patterns = _stable_failure_patterns(prepared["feedback"], _predict(prepared["feedback"], gate), gate.threshold) if round_index < iterations else []
        versions.append({"round": round_index, "signals": sorted(active_signals), "policy": active_policy, "features": len(names), "acceptance": acceptance, "patterns": patterns})
        if not patterns:
            mutations.append({"round": round_index + 1, "added": [], "accepted": False, "reason": "no_stable_pattern"})
            continue
        candidate = set(active_signals)
        for pattern in patterns:
            candidate.update(pattern["candidate_signals"])
        added = sorted(candidate - active_signals)
        if not added:
            mutations.append({"round": round_index + 1, "added": [], "accepted": False, "reason": "no_new_observed_signal"})
            continue
        candidate_prepared = {name: add_policy_features(rows, candidate) for name, rows in splits.items()}
        candidate_names = _feature_names(candidate_prepared["train"] + candidate_prepared["feedback"], candidate)
        candidate_gate = _train_task_conditional_gate(
            candidate_prepared["train"],
            candidate_prepared["feedback"],
            candidate,
            alpha=alpha,
            budget_fraction=budget_fraction,
            epochs=epochs,
        )
        candidate_acceptance = _report(candidate_prepared["acceptance"], candidate_gate, budget_fraction)
        predecessor_risk = float(acceptance["fixed_budget"]["risk"])
        candidate_risk = float(candidate_acceptance["fixed_budget"]["risk"])
        predecessor_ids = _top_budget_ids(prepared["acceptance"], _predict(prepared["acceptance"], gate), budget_fraction)
        candidate_ids = _top_budget_ids(candidate_prepared["acceptance"], _predict(candidate_prepared["acceptance"], candidate_gate), budget_fraction)
        rank_overlap = len(predecessor_ids & candidate_ids) / max(1, len(predecessor_ids))
        risk_guard = candidate_risk <= predecessor_risk + 0.01
        if predecessor_risk <= alpha:
            risk_guard = risk_guard and candidate_risk <= alpha
        accepted = _score(candidate_acceptance) > _score(acceptance) + 1e-4 and risk_guard
        mutations.append({"round": round_index + 1, "added": added, "policy": "task_conditional_score_heads", "task_heads": sorted(candidate_gate.task_gates), "accepted": accepted, "predecessor_score": _score(acceptance), "candidate_score": _score(candidate_acceptance), "predecessor_risk": predecessor_risk, "candidate_risk": candidate_risk, "risk_guard": risk_guard, "top_budget_rank_overlap": rank_overlap, "top_budget_rank_displacement": 1.0 - rank_overlap})
        if accepted:
            active_signals = candidate
            active_policy = "task_conditional_score_heads"
            accepted_rounds.append(round_index + 1)
    final_prepared = {name: add_policy_features(rows, active_signals) for name, rows in splits.items()}
    names = _feature_names(final_prepared["train"] + final_prepared["feedback"], active_signals)
    if active_policy == "task_conditional_score_heads":
        final_gate = _train_task_conditional_gate(final_prepared["train"], final_prepared["feedback"], active_signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
    else:
        final_gate = _train_external_gate(final_prepared["train"], final_prepared["feedback"], names, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
    final_test = _report(final_prepared["test"], final_gate, budget_fraction)
    per_task = {}
    for task in sorted({str(row.metadata["task"]) for row in final_prepared["test"]}):
        per_task[task] = _report([row for row in final_prepared["test"] if row.metadata["task"] == task], final_gate, budget_fraction)
    return {"method": "Conditional Action-Worthiness RHI", "fold": fold, "task_counts": task_counts, "split_sizes": {name: len(rows) for name, rows in splits.items()}, "versions": versions, "mutations": mutations, "accepted_rounds": accepted_rounds, "final_signals": sorted(active_signals), "final_policy": active_policy, "final_test": final_test, "final_test_by_task": per_task}


def run_external_fold_suite(roots: dict[str, str | Path], *, folds: int = 5, iterations: int = 3, alpha: float = 0.10, budget_fraction: float = 0.10, epochs: int = 80) -> dict[str, Any]:
    available = None
    for root in roots.values():
        root = Path(root)
        fold_ids = set()
        for path in (root / "folds").glob("fold_*") if (root / "folds").exists() else root.glob("fold_*"):
            try:
                fold_ids.add(int(path.name.removeprefix("fold_")))
            except ValueError:
                continue
        available = fold_ids if available is None else available & fold_ids
    common_folds = sorted(available or set())[:folds]
    if not common_folds:
        raise ValueError("the supplied external roots have no common fold")
    results = [run_external_three_task_experiment(roots, fold=fold, iterations=iterations, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs) for fold in common_folds]
    static_results = []
    for fold in common_folds:
        signals = set(ALL_CANDIDATE_SIGNALS)
        loaded = {name: load_external_split_records(root, fold=fold) for name, root in roots.items()}
        rows = {split: [item for name in loaded for item in loaded[name][split]] for split in ("train", "feedback", "acceptance", "test")}
        prepared = {split: add_policy_features(items, signals) for split, items in rows.items()}
        names = _feature_names(prepared["train"] + prepared["feedback"], signals)
        gate = train_gate_with_features(prepared["train"], prepared["feedback"], feature_names=names, alpha=alpha, epochs=epochs, learning_rate=0.08, l2=0.01, min_coverage=budget_fraction, balance_benchmarks=True)
        static_results.append(_report(prepared["test"], gate, budget_fraction))
    def avg_reports(reports: list[dict[str, Any]], key: str) -> float:
        return sum(float(report["fixed_budget"].get(key, 0.0)) for report in reports) / len(reports)
    summary = {
        "method": "Conditional Action-Worthiness RHI",
        "folds": common_folds,
        "results": results,
        "aggregate": {
            "rhi_budget_risk_mean": avg_reports([result["final_test"] for result in results], "risk"),
            "rhi_budget_coverage_mean": avg_reports([result["final_test"] for result in results], "coverage"),
            "rhi_budget_hit_rate_mean": avg_reports([result["final_test"] for result in results], "hit_rate"),
            "static_full_budget_risk_mean": avg_reports(static_results, "risk"),
            "static_full_budget_coverage_mean": avg_reports(static_results, "coverage"),
            "static_full_budget_hit_rate_mean": avg_reports(static_results, "hit_rate"),
            "rhi_threshold_risk_mean": sum(float(result["final_test"]["metrics"]["selective_risk"]) for result in results) / len(results),
            "static_full_threshold_risk_mean": sum(float(report["metrics"]["selective_risk"]) for report in static_results) / len(static_results),
            "accepted_mutation_rate": sum(len(result["accepted_rounds"]) > 1 for result in results) / len(results),
        },
    }
    return summary


def _split_trajectories(records: list[ActionRecord], seed: int) -> dict[str, list[ActionRecord]]:
    groups: dict[str, list[ActionRecord]] = defaultdict(list)
    for record in records:
        groups[str(record.metadata["trajectory_id"])].append(record)
    names = ("train", "feedback", "acceptance", "test")
    output = {name: [] for name in names}
    by_task: dict[str, list[str]] = defaultdict(list)
    for trajectory_id, rows in groups.items():
        by_task[str(rows[0].metadata["task"])].append(trajectory_id)
    rng = random.Random(seed)
    for task, task_ids in sorted(by_task.items()):
        rng.shuffle(task_ids)
        n = len(task_ids)
        if n < 4:
            raise ValueError(f"task {task} needs at least four trajectories")
        n_train = max(1, int(round(n * 0.60)))
        n_feedback = max(1, int(round(n * 0.15)))
        n_acceptance = max(1, int(round(n * 0.10)))
        if n_train + n_feedback + n_acceptance >= n:
            n_train, n_feedback, n_acceptance = n - 3, 1, 1
        boundaries = (n_train, n_train + n_feedback, n_train + n_feedback + n_acceptance)
        for index, trajectory_id in enumerate(task_ids):
            bucket = 0 if index < boundaries[0] else 1 if index < boundaries[1] else 2 if index < boundaries[2] else 3
            output[names[bucket]].extend(groups[trajectory_id])
    if any(not output[name] for name in names):
        raise ValueError(f"trajectory split is too small: { {name: len(rows) for name, rows in output.items()} }")
    return output


def feature_policy(task: str, stage: str, active_signals: set[str]) -> list[str]:
    selected = {name for name in active_signals if "::" not in name}
    selected.update(name.split("::", 1)[1] for name in active_signals if name.startswith(f"{task}::"))
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
        task_specific = {name.split("::", 1)[1] for name in active_signals if name.startswith(f"{task}::")}
        global_signals = {name for name in active_signals if "::" not in name}
        for name in global_signals | task_specific:
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


def _predict(records: list[ActionRecord], gate: TrainedGate | TaskConditionalGate) -> list[float]:
    return gate.predict_proba(records)


def _train_external_gate(
    train_records: list[ActionRecord],
    feedback_records: list[ActionRecord],
    feature_names: list[str],
    *,
    alpha: float,
    budget_fraction: float,
    epochs: int,
) -> TrainedGate:
    gate = train_gate_with_features(
        train_records,
        feedback_records,
        feature_names=feature_names,
        alpha=alpha,
        epochs=epochs,
        learning_rate=0.08,
        l2=0.01,
        min_coverage=budget_fraction,
        balance_benchmarks=True,
    )
    probabilities = gate.predict_proba(feedback_records)
    labels = [record.label for record in feedback_records]
    groups = [str(record.metadata.get("task", record.benchmark)) for record in feedback_records]
    calibration = threshold_for_grouped_selective_risk(
        labels,
        probabilities,
        groups,
        alpha=alpha,
        min_coverage=budget_fraction,
    )
    gate.threshold = float(calibration["threshold"])
    gate.calibration = {str(key): float(value) for key, value in calibration.items()}
    return gate


def _train_task_conditional_gate(
    train_records: list[ActionRecord],
    feedback_records: list[ActionRecord],
    active_signals: set[str],
    *,
    alpha: float,
    budget_fraction: float,
    epochs: int,
) -> TaskConditionalGate:
    fallback_names = _feature_names(train_records + feedback_records, active_signals)
    fallback = _train_external_gate(
        train_records,
        feedback_records,
        fallback_names,
        alpha=alpha,
        budget_fraction=budget_fraction,
        epochs=epochs,
    )
    task_gates: dict[str, TrainedGate] = {}
    tasks = sorted({str(record.metadata.get("task", record.benchmark)) for record in train_records + feedback_records})
    for task in tasks:
        task_train = [record for record in train_records if str(record.metadata.get("task", record.benchmark)) == task]
        task_feedback = [record for record in feedback_records if str(record.metadata.get("task", record.benchmark)) == task]
        if len(task_train) < 8 or len(task_feedback) < 8:
            continue
        names = _feature_names(task_train + task_feedback, active_signals)
        task_gates[task] = train_gate_with_features(
            task_train,
            task_feedback,
            feature_names=names,
            alpha=alpha,
            epochs=epochs,
            learning_rate=0.08,
            l2=0.01,
            min_coverage=budget_fraction,
            balance_benchmarks=False,
        )
    return TaskConditionalGate(fallback, task_gates)


def _report(records: list[ActionRecord], gate: TrainedGate | TaskConditionalGate, budget_fraction: float) -> dict[str, Any]:
    probabilities = _predict(records, gate)
    labels = [record.label for record in records]
    budget = max(1, int(len(records) * budget_fraction))
    metrics = binary_metrics(labels, probabilities, gate.threshold)
    fixed = _fixed_budget_report(labels, [record.utility for record in records], probabilities, budget_fraction)
    gain = discovery_gain(labels, [record.utility for record in records], probabilities, budget)
    by_task = {}
    for task in sorted({str(record.metadata.get("task", record.benchmark)) for record in records}):
        task_records = [record for record in records if str(record.metadata.get("task", record.benchmark)) == task]
        task_probabilities = _predict(task_records, gate)
        task_labels = [record.label for record in task_records]
        task_budget = max(1, int(len(task_records) * budget_fraction))
        by_task[task] = {
            "metrics": binary_metrics(task_labels, task_probabilities, gate.threshold),
            "discovery_gain": discovery_gain(task_labels, [record.utility for record in task_records], task_probabilities, task_budget),
            "n": len(task_records),
        }
    if by_task:
        macro_metrics = {
            key: sum(float(item["metrics"].get(key, 0.0)) for item in by_task.values()) / len(by_task)
            for key in ("coverage", "selective_risk", "ece", "brier", "aurc")
        }
        macro_gain = {
            key: sum(float(item["discovery_gain"].get(key, 0.0)) for item in by_task.values()) / len(by_task)
            for key in ("hit_efficiency", "utility_efficiency", "hit_rate", "mean_utility")
        }
    else:
        macro_metrics, macro_gain = {}, {}
    return {"metrics": metrics, "fixed_budget": fixed, "discovery_gain": gain, "macro_metrics": macro_metrics, "macro_discovery_gain": macro_gain, "by_task": by_task, "threshold": gate.threshold, "n": len(records)}


def _fixed_budget_report(labels: list[int], utilities: list[float], probabilities: list[float], budget_fraction: float) -> dict[str, float]:
    if not labels:
        return {"budget": 0.0, "coverage": 0.0, "risk": 0.0, "hit_rate": 0.0, "mean_utility": 0.0}
    budget = max(1, min(len(labels), math.ceil(len(labels) * budget_fraction)))
    selected = sorted(range(len(labels)), key=lambda index: probabilities[index], reverse=True)[:budget]
    return {
        "budget": float(budget),
        "coverage": budget / len(labels),
        "risk": sum(1 - labels[index] for index in selected) / budget,
        "hit_rate": sum(labels[index] for index in selected) / budget,
        "mean_utility": sum(utilities[index] for index in selected) / budget,
    }


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
        candidate_signals = []
        for signal in TASK_SIGNAL_PRIORS.get(task, ()):
            values = [records[i].features.get(signal, 0.0) for i in indices]
            if signal not in {"cost", "reversibility", "verbal_confidence"} and values and max(values) - min(values) > 1e-6:
                candidate_signals.append(f"{task}::{signal}")
        patterns.append({"task": task, "stage": stage, "n": len(indices), "mean_abs_error": group_error, "global_mean_abs_error": global_error, "threshold": threshold, "candidate_signals": candidate_signals})
    return patterns


def _score(report: dict[str, Any]) -> float:
    metrics = report.get("macro_metrics", report["metrics"])
    gain = report.get("macro_discovery_gain", report["discovery_gain"])
    fixed = report.get("fixed_budget", {})
    return 0.45 * gain.get("utility_efficiency", 0.0) + 0.25 * gain.get("hit_efficiency", 0.0) + 0.20 * (1.0 - fixed.get("risk", metrics.get("selective_risk", 1.0))) + 0.10 * (1.0 - metrics.get("brier", 1.0))


def run_conditional_action_rhi(records: list[ActionRecord], *, iterations: int = 3, seed: int = 1729, alpha: float = 0.10, budget_fraction: float = 0.10, epochs: int = 120) -> dict[str, Any]:
    splits = _split_trajectories(records, seed)
    active_signals = {"verbal_confidence", "cost", "reversibility"}
    active_policy = "global_score"
    versions = []
    patterns_by_round = []
    accepted_rounds = [0]
    mutations: list[dict[str, Any]] = []
    final_gate = None
    for round_index in range(iterations + 1):
        prepared = {name: add_policy_features(rows, active_signals) for name, rows in splits.items()}
        names = _feature_names(prepared["train"] + prepared["feedback"], active_signals)
        gate = (
            _train_task_conditional_gate(prepared["train"], prepared["feedback"], active_signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
            if active_policy == "task_conditional_score_heads"
            else train_gate_with_features(prepared["train"], prepared["feedback"], feature_names=names, alpha=alpha, epochs=epochs, learning_rate=0.08, l2=0.01, min_coverage=budget_fraction)
        )
        acceptance = _report(prepared["acceptance"], gate, budget_fraction)
        feedback_probabilities = _predict(prepared["feedback"], gate)
        patterns = _stable_failure_patterns(prepared["feedback"], feedback_probabilities, gate.threshold) if round_index < iterations else []
        versions.append({"round": round_index, "active_signals": sorted(active_signals), "policy": active_policy, "feature_count": len(names), "acceptance": acceptance, "patterns": patterns})
        final_gate = gate
        patterns_by_round.append(patterns)
        if not patterns:
            mutations.append({"round": round_index + 1, "candidate_signals": [], "accepted": False, "reason": "no_stable_pattern"})
            continue
        candidate = set(active_signals)
        for pattern in patterns:
            candidate.update(name for name in pattern["candidate_signals"] if name not in active_signals)
        if candidate == active_signals:
            mutations.append({"round": round_index + 1, "candidate_signals": [], "accepted": False, "reason": "no_new_observed_signal"})
            continue
        candidate_prepared = {name: add_policy_features(rows, candidate) for name, rows in splits.items()}
        candidate_names = _feature_names(candidate_prepared["train"] + candidate_prepared["feedback"], candidate)
        candidate_gate = _train_task_conditional_gate(candidate_prepared["train"], candidate_prepared["feedback"], candidate, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
        candidate_acceptance = _report(candidate_prepared["acceptance"], candidate_gate, budget_fraction)
        predecessor_ids = _top_budget_ids(prepared["acceptance"], _predict(prepared["acceptance"], gate), budget_fraction)
        candidate_ids = _top_budget_ids(candidate_prepared["acceptance"], _predict(candidate_prepared["acceptance"], candidate_gate), budget_fraction)
        rank_overlap = len(predecessor_ids & candidate_ids) / max(1, len(predecessor_ids))
        accepted = _score(candidate_acceptance) >= _score(acceptance) and candidate_acceptance["metrics"]["selective_risk"] <= alpha + 0.05
        mutations.append({"round": round_index + 1, "candidate_signals": sorted(candidate - active_signals), "policy": "task_conditional_score_heads", "accepted": accepted, "predecessor_score": _score(acceptance), "candidate_score": _score(candidate_acceptance), "predecessor_risk": acceptance["metrics"]["selective_risk"], "candidate_risk": candidate_acceptance["metrics"]["selective_risk"], "top_budget_rank_overlap": rank_overlap, "top_budget_rank_displacement": 1.0 - rank_overlap})
        if accepted:
            active_signals = candidate
            active_policy = "task_conditional_score_heads"
            accepted_rounds.append(round_index + 1)
    final_prepared = {name: add_policy_features(rows, active_signals) for name, rows in splits.items()}
    final_names = _feature_names(final_prepared["train"] + final_prepared["feedback"], active_signals)
    final_gate = (
        _train_task_conditional_gate(final_prepared["train"], final_prepared["feedback"], active_signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
        if active_policy == "task_conditional_score_heads"
        else train_gate_with_features(final_prepared["train"], final_prepared["feedback"], feature_names=final_names, alpha=alpha, epochs=epochs, learning_rate=0.08, l2=0.01, min_coverage=budget_fraction)
    )
    final_test = _report(final_prepared["test"], final_gate, budget_fraction)
    for version in versions:
        version["accepted"] = version["round"] in accepted_rounds
    by_task = {}
    for task in sorted({str(row.metadata["task"]) for row in final_prepared["test"]}):
        task_rows = [row for row in final_prepared["test"] if str(row.metadata["task"]) == task]
        by_task[task] = _report(task_rows, final_gate, budget_fraction)
    return {"method": "Conditional Action-Worthiness RHI", "single_agent": True, "split_sizes": {name: len(rows) for name, rows in splits.items()}, "iterations": iterations, "budget_fraction": budget_fraction, "alpha": alpha, "versions": versions, "stable_failure_patterns": patterns_by_round, "mutations": mutations, "accepted_rounds": accepted_rounds, "final_test": final_test, "final_test_by_task": by_task, "final_signals": sorted(active_signals), "final_policy": active_policy}


def run_experiment(input_path: str | Path, out_dir: str | Path, *, seed: int = 1729, iterations: int = 3, epochs: int = 120) -> dict[str, Any]:
    records = load_action_records(input_path)
    result = run_conditional_action_rhi(records, seed=seed, iterations=iterations, epochs=epochs)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final_metrics = result["final_test"]["metrics"]
    lines = ["# Conditional Action-Worthiness RHI", "", f"- Single agent action units: {result['single_agent']}", f"- Split sizes: {result['split_sizes']}", f"- Accepted rounds: {result['accepted_rounds']}", f"- Final active signals: {', '.join(result['final_signals'])}", "- Test is evaluated only after acceptance; benchmark labels are offline proxies.", "", "| Round | Features | Acceptance score | Accepted |", "|---:|---:|---:|:---:|"]
    for version in result["versions"]:
        lines.append(f"| {version['round']} | {version['feature_count']} | {_score(version['acceptance']):.4f} | {version['accepted']} |")
    lines += ["", "## Mutations", "", "| Round | Added signals | Accepted |", "|---:|---|:---:|"]
    for mutation in result["mutations"]:
        lines.append(f"| {mutation['round']} | {', '.join(mutation['candidate_signals']) or '-'} | {mutation['accepted']} |")
    lines += ["", f"Final test: risk={final_metrics['selective_risk']:.4f}, coverage={final_metrics['coverage']:.4f}, ECE={final_metrics['ece']:.4f}, Brier={final_metrics['brier']:.4f}.", "", "## Per-task test", "", "| Task | Risk | Coverage | ECE |", "|---|---:|---:|---:|"]
    for task, report in result["final_test_by_task"].items():
        metrics = report["metrics"]
        lines.append(f"| {task} | {metrics['selective_risk']:.4f} | {metrics['coverage']:.4f} | {metrics['ece']:.4f} |")
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_seed_suite(input_path: str | Path, out_dir: str | Path, *, seeds: list[int], iterations: int = 3, epochs: int = 80) -> dict[str, Any]:
    records = load_action_records(input_path)
    rows = []
    for seed in seeds:
        result = run_conditional_action_rhi(records, seed=seed, iterations=iterations, epochs=epochs)
        splits = _split_trajectories(records, seed)
        static_signals = set(ALL_CANDIDATE_SIGNALS)
        static_prepared = {name: add_policy_features(value, static_signals) for name, value in splits.items()}
        static_names = _feature_names(static_prepared["train"] + static_prepared["feedback"], static_signals)
        static_gate = train_gate_with_features(
            static_prepared["train"],
            static_prepared["feedback"],
            feature_names=static_names,
            alpha=0.10,
            epochs=epochs,
            learning_rate=0.08,
            l2=0.01,
            min_coverage=0.10,
        )
        static_test = _report(static_prepared["test"], static_gate, 0.10)
        rows.append({
            "seed": seed,
            "accepted_rounds": result["accepted_rounds"],
            "final_signals": result["final_signals"],
            "final_test": result["final_test"],
            "final_test_by_task": result["final_test_by_task"],
            "mutations": result["mutations"],
            "static_full_test": static_test,
        })
    def mean(path: str) -> float:
        return sum(float(row["final_test"]["metrics"][path]) for row in rows) / len(rows)
    summary = {
        "method": "Conditional Action-Worthiness RHI",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "rows": rows,
        "aggregate": {
            "risk_mean": mean("selective_risk"),
            "risk_std": math.sqrt(sum((float(row["final_test"]["metrics"]["selective_risk"]) - mean("selective_risk")) ** 2 for row in rows) / len(rows)),
            "coverage_mean": mean("coverage"),
            "ece_mean": mean("ece"),
            "brier_mean": mean("brier"),
            "accepted_mutation_rate": sum(len(row["accepted_rounds"]) > 1 for row in rows) / len(rows),
            "static_full_risk_mean": sum(float(row["static_full_test"]["metrics"]["selective_risk"]) for row in rows) / len(rows),
            "static_full_coverage_mean": sum(float(row["static_full_test"]["metrics"]["coverage"]) for row in rows) / len(rows),
            "static_full_ece_mean": sum(float(row["static_full_test"]["metrics"]["ece"]) for row in rows) / len(rows),
        },
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "seed_suite.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Conditional Action-Worthiness RHI: five-seed suite",
        "",
        "All methods use the same task-stratified trajectory split and 10% action budget.",
        "",
        "| Method | Risk | Coverage | ECE | Brier | Accepted mutation rate |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Conditional RHI | {summary['aggregate']['risk_mean']:.4f} | {summary['aggregate']['coverage_mean']:.4f} | {summary['aggregate']['ece_mean']:.4f} | {summary['aggregate']['brier_mean']:.4f} | {summary['aggregate']['accepted_mutation_rate']:.2f} |",
        f"| Static full | {summary['aggregate']['static_full_risk_mean']:.4f} | {summary['aggregate']['static_full_coverage_mean']:.4f} | {summary['aggregate']['static_full_ece_mean']:.4f} | - | - |",
        "",
        "The conditional RHI mutation rate is reported rather than assuming that every proposed mutation is beneficial.",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-agent conditional action-worthiness RHI.")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", default="", help="comma-separated seeds for a multi-seed suite")
    args = parser.parse_args()
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        result = run_seed_suite(args.actions, args.out_dir, seeds=seeds, iterations=args.iterations, epochs=min(args.epochs, 80))
        print(json.dumps({"seeds": result["seeds"], "aggregate": result["aggregate"]}, ensure_ascii=False))
    else:
        result = run_experiment(args.actions, args.out_dir, seed=args.seed, iterations=args.iterations, epochs=args.epochs)
        print(json.dumps({"split_sizes": result["split_sizes"], "final_signals": result["final_signals"], "versions": result["versions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
