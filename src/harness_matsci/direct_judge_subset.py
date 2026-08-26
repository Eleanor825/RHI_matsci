from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .calibration import threshold_for_selective_risk
from .direct_judge import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DirectJudgeError, LLMDirectJudge
from .io import read_jsonl, write_json, write_jsonl
from .metrics import action_worthiness_score, binary_metrics, discovery_gain, scientific_discovery_metrics
from .schema import ActionRecord
from .training import evidence_heuristic_scores, verbal_confidence_scores
from .voi import VOI_SEED_HARNESS, evaluate_voi, fit_voi_model, train_voi_rhi


@dataclass(frozen=True)
class DirectJudgeSubsetConfig:
    records_path: str
    out_dir: str
    judge_cache_path: str
    seed: int = 20260816
    validation_fraction: float = 0.5
    alpha: float = 0.1
    budget_fraction: float = 0.1
    epochs: int = 80
    learning_rate: float = 0.08
    l2: float = 0.01
    rhi_iterations: int = 3
    cost_weight: float = 0.15


@dataclass(frozen=True)
class DirectJudgeScoringConfig:
    records_path: str
    cache_path: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    timeout: float = 90.0
    max_retries: int = 3
    max_workers: int = 4
    limit: int | None = None
    allow_failures: bool = False


def split_records_file(
    records_path: str | Path,
    out_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    seed: int = 20260816,
    validation_fraction: float = 0.5,
) -> dict[str, Any]:
    records = read_jsonl(records_path)
    split_records = assign_validation_test_splits(records, seed=seed, validation_fraction=validation_fraction)
    write_jsonl(split_records, out_path)
    summary = _split_summary(split_records, seed=seed, validation_fraction=validation_fraction)
    if summary_path:
        write_json(summary, summary_path)
    return summary


def assign_validation_test_splits(
    records: list[ActionRecord],
    *,
    seed: int = 20260816,
    validation_fraction: float = 0.5,
) -> list[ActionRecord]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    buckets: dict[tuple[str, int], list[ActionRecord]] = {}
    for record in records:
        buckets.setdefault((record.benchmark, record.label), []).append(record)
    result: list[ActionRecord] = []
    for key, bucket in sorted(buckets.items()):
        ordered = sorted(bucket, key=lambda record: _hash_float(seed, "split", key[0], key[1], record.record_id))
        validation_count = int(round(len(ordered) * validation_fraction))
        if len(ordered) > 1:
            validation_count = min(len(ordered) - 1, max(1, validation_count))
        for index, record in enumerate(ordered):
            split = "validation" if index < validation_count else "test"
            result.append(_with_split(record, split))
    return sorted(result, key=lambda record: record.record_id)


def score_direct_judge_records(config: DirectJudgeScoringConfig, *, api_key: str | None = None) -> dict[str, Any]:
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        raise ValueError("set OPENAI_API_KEY before scoring direct judge records")
    records = read_jsonl(config.records_path)
    if config.limit is not None:
        records = records[: max(0, config.limit)]
    judge = LLMDirectJudge(
        model=config.model,
        api_key=resolved_key,
        base_url=config.base_url,
        reasoning_effort=config.reasoning_effort,
        cache_path=config.cache_path,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )
    missing = [record for record in records if record.record_id not in judge._cache]
    started = time.time()
    failures: dict[str, str] = {}
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as executor:
            futures = {executor.submit(judge._score_record, record): record for record in missing}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    judge._cache[record.record_id] = float(future.result())
                    judge._save_cache()
                except Exception as error:
                    failures[record.record_id] = str(error)
                    if not config.allow_failures:
                        judge._save_cache()
                        raise DirectJudgeError(
                            f"direct judge scoring failed after caching {len(judge._cache)} scores; first failure={record.record_id}: {error}"
                        ) from error
    judge._save_cache()
    summary = {
        "schema": "direct-judge-parallel-score-v1",
        "config": asdict(config),
        "records": len(records),
        "cached_before_or_scored": len([record for record in records if record.record_id in judge._cache]),
        "newly_requested": len(missing),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    if failures and not config.allow_failures:
        raise DirectJudgeError(f"direct judge scoring has {len(failures)} failures")
    return summary


def evaluate_direct_judge_subset(config: DirectJudgeSubsetConfig) -> dict[str, Any]:
    records = read_jsonl(config.records_path)
    scores = _load_scores(config.judge_cache_path)
    missing = [record.record_id for record in records if record.record_id not in scores]
    if missing:
        raise ValueError(f"direct judge cache is missing {len(missing)} scores; first missing={missing[0]}")
    validation_records = [record for record in records if record.split == "validation"]
    test_records = [record for record in records if record.split == "test"]
    if not validation_records or not test_records:
        raise ValueError("records must contain non-empty validation and test splits")
    train_records, feedback_records, acceptance_records = _training_partitions(validation_records, seed=config.seed)
    signal_methods = {
        "llm_direct_judge": lambda rows: [scores[record.record_id] for record in rows],
        "verbal_confidence": verbal_confidence_scores,
        "evidence_heuristic": evidence_heuristic_scores,
    }
    methods: dict[str, dict[str, Any]] = {}
    for name, score_fn in signal_methods.items():
        methods[name] = _evaluate_score_signal(
            validation_records,
            test_records,
            score_fn(validation_records),
            score_fn(test_records),
            alpha=config.alpha,
            budget_fraction=config.budget_fraction,
            cost_weight=config.cost_weight,
        )
    local_model = fit_voi_model(
        train_records,
        feedback_records,
        _subset_voi_harness(),
        alpha=config.alpha,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
    )
    methods["local_voi_harness"] = _compact_voi_evaluation(
        evaluate_voi(test_records, local_model, budget_fraction=config.budget_fraction)
    )
    rhi = train_voi_rhi(
        train_records,
        feedback_records,
        acceptance_records,
        test_records,
        iterations=config.rhi_iterations,
        seed=config.seed,
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
    )
    methods["scivoi_rhi"] = _compact_voi_evaluation(rhi["test"])
    methods["scivoi_rhi"]["accepted_versions"] = list(rhi["accepted_versions"])
    methods["scivoi_rhi"]["n_accepted"] = max(0, len(rhi["accepted_versions"]) - 1)
    report = {
        "schema": "direct-judge-main-materials-subset-eval-v1",
        "config": asdict(config),
        "data": _split_summary(records, seed=config.seed, validation_fraction=config.validation_fraction),
        "train_feedback_acceptance_split": {
            "train": len(train_records),
            "feedback": len(feedback_records),
            "acceptance": len(acceptance_records),
            "test": len(test_records),
        },
        "method_summary": methods,
        "best_by_primary_utility": max(methods, key=lambda name: methods[name]["oracle_normalized_net_utility"]),
        "best_by_risk_adjusted_utility": max(methods, key=lambda name: methods[name]["risk_adjusted_utility"]),
        "interpretation": [
            "The direct LLM judge is calibrated on validation records and ranked by one-shot p_success on the held-out test split.",
            "Sci-VoI-RHI uses the same validation records, but recursively mutates the executable harness using train, feedback, and acceptance partitions before touching test.",
            "Primary utility is oracle-normalized net scientific utility at a fixed top-10% action budget; higher is better.",
        ],
    }
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, out_dir / "summary.json")
    (out_dir / "README.md").write_text(render_direct_judge_subset_markdown(report), encoding="utf-8")
    return report


def render_direct_judge_subset_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Main-Materials GPT-5.5 Direct-Judge Subset",
        "",
        "This run evaluates a one-shot LLM-as-judge baseline on the same validation/test subset used by local VoI and Sci-VoI-RHI.",
        "",
        f"- Records: `{report['data']['total_records']}`.",
        f"- Validation/test: `{report['data']['splits'].get('validation', 0)}` / `{report['data']['splits'].get('test', 0)}`.",
        f"- Best primary utility: `{report['best_by_primary_utility']}`.",
        f"- Best risk-adjusted utility: `{report['best_by_risk_adjusted_utility']}`.",
        "",
        "## Results",
        "",
        "| Method | Net utility | Risk-adjusted | Risk | Hit rate | Coverage | Utility efficiency | Diagnostic score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in sorted(report["method_summary"].items()):
        lines.append(
            f"| `{name}` | {values['oracle_normalized_net_utility']:.4f} | {values['risk_adjusted_utility']:.4f} | "
            f"{values['execute_selective_risk']:.4f} | {values['hit_rate']:.4f} | {values['coverage']:.4f} | "
            f"{values.get('utility_efficiency', 0.0):.4f} | {_format_optional(values.get('diagnostic_score'))} |"
        )
    lines.extend([
        "",
        "## Protocol",
        "",
        "- `llm_direct_judge` sees only visible action context, candidate action, and pre-execution evidence; it outputs `p_success` once per action.",
        "- `local_voi_harness` trains a non-recursive local value-of-information harness on the validation partition.",
        "- `scivoi_rhi` uses train/feedback/acceptance partitions from validation data to recursively mutate and accept the harness before test evaluation.",
        "- The held-out test split is used only for final reporting.",
    ])
    return "\n".join(lines) + "\n"


def _evaluate_score_signal(
    validation_records: list[ActionRecord],
    test_records: list[ActionRecord],
    validation_scores: list[float],
    test_scores: list[float],
    *,
    alpha: float,
    budget_fraction: float,
    cost_weight: float,
) -> dict[str, Any]:
    threshold = threshold_for_selective_risk(
        [record.label for record in validation_records],
        validation_scores,
        alpha=alpha,
        min_coverage=0.1,
    )["threshold"]
    evaluation = _evaluate_ranked_scores(
        validation_records,
        test_records,
        test_scores,
        threshold=threshold,
        budget_fraction=budget_fraction,
        cost_weight=cost_weight,
    )
    evaluation["threshold"] = float(threshold)
    return evaluation


def _evaluate_ranked_scores(
    source_records: list[ActionRecord],
    test_records: list[ActionRecord],
    scores: list[float],
    *,
    threshold: float,
    budget_fraction: float,
    cost_weight: float,
) -> dict[str, Any]:
    if len(scores) != len(test_records):
        raise ValueError("score count must match test record count")
    minimum = min(record.utility for record in source_records) if source_records else min(record.utility for record in test_records)
    maximum = max(record.utility for record in source_records) if source_records else max(record.utility for record in test_records)

    def normalize(value: float) -> float:
        if maximum <= minimum:
            return 0.5
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    probs = [max(0.0, min(1.0, float(score))) for score in scores]
    budget = max(1, int(len(test_records) * budget_fraction)) if test_records else 0
    selected = sorted(range(len(test_records)), key=lambda index: probs[index], reverse=True)[:budget]
    utilities = [
        normalize(record.utility) - cost_weight * max(0.0, min(1.0, float(record.features.get("cost", 0.5))))
        for record in test_records
    ]
    outcome_utilities = [
        normalize(record.utility) * float(record.label)
        - cost_weight * max(0.0, min(1.0, float(record.features.get("cost", 0.5))))
        for record in test_records
    ]
    oracle = sorted(range(len(test_records)), key=lambda index: utilities[index], reverse=True)[:budget]
    outcome_oracle = sorted(range(len(test_records)), key=lambda index: outcome_utilities[index], reverse=True)[:budget]
    selected_indices = [index for index, score in enumerate(probs) if score >= threshold]
    wrong = sum(1 - test_records[index].label for index in selected_indices)
    labels = [record.label for record in test_records]
    metrics = binary_metrics(labels, probs, threshold)
    gain = discovery_gain(labels, utilities, probs, budget)
    selected_utility = mean(utilities[index] for index in selected) if selected else 0.0
    oracle_utility = mean(utilities[index] for index in oracle) if oracle else 0.0
    outcome_selected_utility = mean(outcome_utilities[index] for index in selected) if selected else 0.0
    outcome_oracle_utility = mean(outcome_utilities[index] for index in outcome_oracle) if outcome_oracle else 0.0
    selected_regimes = {str(test_records[index].metadata.get("group_id", test_records[index].benchmark)) for index in selected}
    all_regimes = {str(record.metadata.get("group_id", record.benchmark)) for record in test_records}
    risk = wrong / len(selected_indices) if selected_indices else 0.0
    net_utility = selected_utility / oracle_utility if oracle_utility > 0 else 0.0
    return {
        "n_records": len(test_records),
        "budget": budget,
        "oracle_normalized_net_utility": net_utility,
        "risk_adjusted_utility": net_utility * (1.0 - risk),
        "outcome_conditioned_oracle_normalized_net_utility": outcome_selected_utility / outcome_oracle_utility if outcome_oracle_utility > 0 else 0.0,
        "selected_utility": selected_utility,
        "oracle_utility": oracle_utility,
        "hit_rate": mean(test_records[index].label for index in selected) if selected else 0.0,
        "coverage": len(selected_indices) / len(test_records) if test_records else 0.0,
        "execute_selective_risk": risk,
        "confidently_wrong_execute_rate": wrong / len(test_records) if test_records else 0.0,
        "regime_coverage": len(selected_regimes) / len(all_regimes) if all_regimes else 0.0,
        "utility_efficiency": float(gain.get("utility_efficiency", 0.0)),
        "hit_efficiency": float(gain.get("hit_efficiency", 0.0)),
        "diagnostic_score": action_worthiness_score(metrics, gain),
        "metrics": metrics,
        "discovery_gain": gain,
        "scientific_discovery": scientific_discovery_metrics(test_records, probs, budget),
    }


def _compact_voi_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    risk = float(evaluation.get("execute_selective_risk", 0.0))
    net_utility = float(evaluation.get("oracle_normalized_net_utility", 0.0))
    reliability = evaluation.get("reliability", {})
    gain = reliability.get("discovery_gain", {}) if isinstance(reliability, dict) else {}
    metrics = reliability.get("metrics", {}) if isinstance(reliability, dict) else {}
    diagnostic_score = action_worthiness_score(metrics, gain) if metrics and gain else None
    return {
        "n_records": int(evaluation.get("n_records", 0)),
        "budget": int(evaluation.get("budget", 0)),
        "oracle_normalized_net_utility": net_utility,
        "risk_adjusted_utility": net_utility * (1.0 - risk),
        "outcome_conditioned_oracle_normalized_net_utility": float(evaluation.get("outcome_conditioned_oracle_normalized_net_utility", 0.0)),
        "selected_utility": float(evaluation.get("selected_utility", 0.0)),
        "oracle_utility": float(evaluation.get("oracle_utility", 0.0)),
        "hit_rate": float(evaluation.get("hit_rate", 0.0)),
        "coverage": float(evaluation.get("coverage", 0.0)),
        "execute_selective_risk": risk,
        "confidently_wrong_execute_rate": float(evaluation.get("confidently_wrong_execute_rate", 0.0)),
        "verify_rate": float(evaluation.get("verify_rate", 0.0)),
        "stop_rate": float(evaluation.get("stop_rate", 0.0)),
        "regime_coverage": float(evaluation.get("regime_coverage", 0.0)),
        "utility_efficiency": float(gain.get("utility_efficiency", 0.0)) if isinstance(gain, dict) else 0.0,
        "hit_efficiency": float(gain.get("hit_efficiency", 0.0)) if isinstance(gain, dict) else 0.0,
        "diagnostic_score": diagnostic_score,
    }


def _training_partitions(records: list[ActionRecord], *, seed: int) -> tuple[list[ActionRecord], list[ActionRecord], list[ActionRecord]]:
    buckets: dict[tuple[str, int], list[ActionRecord]] = {}
    for record in records:
        buckets.setdefault((record.benchmark, record.label), []).append(record)
    train: list[ActionRecord] = []
    feedback: list[ActionRecord] = []
    acceptance: list[ActionRecord] = []
    for key, bucket in sorted(buckets.items()):
        ordered = sorted(bucket, key=lambda record: _hash_float(seed, "train", key[0], key[1], record.record_id))
        first = max(1, int(len(ordered) * 0.6))
        second = max(first + 1, int(len(ordered) * 0.8)) if len(ordered) > 2 else first
        first = min(first, len(ordered))
        second = min(second, len(ordered))
        train.extend(ordered[:first])
        feedback.extend(ordered[first:second])
        acceptance.extend(ordered[second:])
    if not feedback:
        feedback.append(train.pop())
    if not acceptance:
        acceptance.append(train.pop())
    return train, feedback, acceptance


def _subset_voi_harness() -> dict[str, Any]:
    harness = copy.deepcopy(VOI_SEED_HARNESS)
    harness.update(
        {
            "name": "subset_static_voi",
            "decision_mode": "voi",
            "execute_cost_weight": 0.15,
            "failure_cost_weight": 0.25,
            "epistemic_weight": 0.10,
            "verification_cost_weight": 0.05,
            "verification_uncertainty_floor": 0.03,
            "verification_support_weight": 0.2,
            "min_execute_reliability": 0.0,
            "allow_verification": True,
            "use_cost_signal": True,
            "use_uncertainty_signal": True,
        }
    )
    return harness


def _load_scores(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = payload.get("scores", payload)
    if not isinstance(scores, dict):
        raise ValueError("score cache must be a JSON object with a scores field")
    return {str(key): float(value) for key, value in scores.items() if _is_finite_number(value)}


def _split_summary(records: list[ActionRecord], *, seed: int, validation_fraction: float) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        task = by_task.setdefault(record.benchmark, {"records": 0, "positive_records": 0, "splits": {}})
        task["records"] += 1
        task["positive_records"] += int(record.label)
        task["splits"][record.split] = task["splits"].get(record.split, 0) + 1
    for task in by_task.values():
        task["positive_rate"] = task["positive_records"] / task["records"] if task["records"] else 0.0
    return {
        "schema": "direct-judge-subset-split-v1",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "total_records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "tasks": dict(sorted(by_task.items())),
    }


def _with_split(record: ActionRecord, split: str) -> ActionRecord:
    payload = record.to_json()
    payload["split"] = split
    return ActionRecord.from_json(payload)


def _hash_float(*parts: object) -> float:
    digest = sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _format_optional(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare, score, and evaluate a direct LLM judge subset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--records", required=True)
    split_parser.add_argument("--out", required=True)
    split_parser.add_argument("--summary-out")
    split_parser.add_argument("--seed", type=int, default=20260816)
    split_parser.add_argument("--validation-fraction", type=float, default=0.5)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--records", required=True)
    score_parser.add_argument("--cache", required=True)
    score_parser.add_argument("--model", default=DEFAULT_MODEL)
    score_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    score_parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    score_parser.add_argument("--timeout", type=float, default=90.0)
    score_parser.add_argument("--retries", type=int, default=3)
    score_parser.add_argument("--workers", type=int, default=4)
    score_parser.add_argument("--limit", type=int)
    score_parser.add_argument("--allow-failures", action="store_true")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--records", required=True)
    eval_parser.add_argument("--cache", required=True)
    eval_parser.add_argument("--out-dir", required=True)
    eval_parser.add_argument("--seed", type=int, default=20260816)
    eval_parser.add_argument("--validation-fraction", type=float, default=0.5)
    eval_parser.add_argument("--alpha", type=float, default=0.1)
    eval_parser.add_argument("--budget-fraction", type=float, default=0.1)
    eval_parser.add_argument("--epochs", type=int, default=80)
    eval_parser.add_argument("--learning-rate", type=float, default=0.08)
    eval_parser.add_argument("--l2", type=float, default=0.01)
    eval_parser.add_argument("--rhi-iterations", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "split":
        summary = split_records_file(
            args.records,
            args.out,
            summary_path=args.summary_out,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "score":
        summary = score_direct_judge_records(
            DirectJudgeScoringConfig(
                records_path=args.records,
                cache_path=args.cache,
                model=args.model,
                base_url=args.base_url,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
                max_retries=args.retries,
                max_workers=args.workers,
                limit=args.limit,
                allow_failures=args.allow_failures,
            )
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    report = evaluate_direct_judge_subset(
        DirectJudgeSubsetConfig(
            records_path=args.records,
            judge_cache_path=args.cache,
            out_dir=args.out_dir,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            alpha=args.alpha,
            budget_fraction=args.budget_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            rhi_iterations=args.rhi_iterations,
        )
    )
    print(f"wrote direct judge subset evaluation to {args.out_dir}; best={report['best_by_primary_utility']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
