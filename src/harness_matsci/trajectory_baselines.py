from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable

from .schema import ActionRecord
from .training import evidence_heuristic_scores


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


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _record(row: dict[str, Any]) -> ActionRecord:
    return ActionRecord.from_json(row["source_action_record"])


def _top_fraction(scores: list[float], fraction: float) -> set[int]:
    count = max(1, int(len(scores) * fraction))
    return set(sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:count])


def _metrics(rows: list[dict[str, Any]], selected: set[int]) -> dict[str, Any]:
    executed = [rows[index] for index in sorted(selected)]
    labels = [int(row["outcome"]["label"]) for row in executed]
    utilities = [float(row["outcome"]["utility"]) for row in executed]
    costs = [float(row.get("cost", 0.0)) for row in executed]
    successes = sum(labels)
    gross = statistics.mean(utilities) if utilities else 0.0
    mean_cost = statistics.mean(costs) if costs else 0.0
    return {
        "n_trajectories": len(rows),
        "n_selected": len(executed),
        "coverage": len(executed) / len(rows) if rows else 0.0,
        "selective_risk": 1.0 - successes / len(executed) if executed else 0.0,
        "hit_rate": successes / len(executed) if executed else 0.0,
        "mean_utility_selected": gross,
        "mean_cost_selected": mean_cost,
        "net_utility_selected": gross - 0.15 * mean_cost,
        "total_utility_per_trajectory": sum(utilities) / len(rows) if rows else 0.0,
        "total_net_utility_per_trajectory": (sum(utilities) - 0.15 * sum(costs)) / len(rows) if rows else 0.0,
    }


def evaluate_prompt_baselines(
    trajectory_path: str | Path,
    out_dir: str | Path,
    *,
    budget_fraction: float = 0.10,
    seed: int = 1729,
) -> dict[str, Any]:
    rows = _rows(trajectory_path)
    records = [_record(row) for row in rows]
    by_task = {task: [index for index, row in enumerate(rows) if row["task"] == task] for task in sorted({row["task"] for row in rows})}
    score_functions: dict[str, Callable[[ActionRecord], float]] = {
        "verbal_confidence": lambda record: record.features.get("verbal_confidence", 0.5),
        "evidence_heuristic": lambda record: evidence_heuristic_scores([record])[0],
        "static_reliability": _reliability,
    }
    policies: dict[str, dict[str, Any]] = {}
    policies["always_execute"] = {"kind": "all", "selected": set(range(len(rows)))}
    rng = random.Random(seed)
    policies["random_top10"] = {"kind": "budget", "selected": set(rng.sample(range(len(rows)), max(1, int(len(rows) * budget_fraction))))}
    for name, score_fn in score_functions.items():
        selected: set[int] = set()
        for indices in by_task.values():
            scores = [score_fn(records[index]) for index in indices]
            selected.update(indices[index] for index in _top_fraction(scores, budget_fraction))
        policies[f"{name}_top10"] = {"kind": "budget", "selected": selected}
    prompt_selected = {
        index for index, row in enumerate(rows) if row["terminal_action"] == "execute_candidate"
    }
    policies["prompt_policy"] = {"kind": "trajectory", "selected": prompt_selected}
    oracle_selected: set[int] = set()
    for indices in by_task.values():
        scores = [float(rows[index]["outcome"]["utility"]) for index in indices]
        oracle_selected.update(indices[index] for index in _top_fraction(scores, budget_fraction))
    policies["oracle_top10_reference"] = {"kind": "oracle_reference", "selected": oracle_selected}

    results: dict[str, Any] = {}
    for name, policy in policies.items():
        result = _metrics(rows, policy["selected"])
        result["kind"] = policy["kind"]
        result["by_task"] = {}
        for task, indices in by_task.items():
            subset_rows = [rows[index] for index in indices]
            local_selected = {position for position, index in enumerate(indices) if index in policy["selected"]}
            result["by_task"][task] = _metrics(subset_rows, local_selected)
        results[name] = result
    report = {
        "schema_version": "prompt-trajectory-baselines-v1",
        "trajectory_path": str(trajectory_path),
        "n_trajectories": len(rows),
        "budget_fraction": budget_fraction,
        "seed": seed,
        "results": results,
        "direct_llm_judge": {"status": "pending_provider_run", "included": False},
        "scivoi_rhi": {"status": "pending_trajectory_policy_replay", "included": False},
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baseline_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (destination / "BASELINE_RESULTS.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Prompt-Trajectory Baselines",
        "",
        f"Same fixed trajectory set: `{report['n_trajectories']}` records; budget: `{report['budget_fraction']:.0%}` per task for ranking baselines.",
        "",
        "| Method | Coverage | Risk ↓ | Hit rate ↑ | Net utility/action ↑ | Total net utility/trajectory ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["results"].items():
        lines.append(
            f"| `{name}` | {result['coverage']:.3f} | {result['selective_risk']:.3f} | {result['hit_rate']:.3f} | {result['net_utility_selected']:.4f} | {result['total_net_utility_per_trajectory']:.4f} |"
        )
    lines += [
        "",
        "`oracle_top10_reference` is an upper-bound reference and is not an executable baseline.",
        "`prompt_policy` is the deterministic prompt-conditioned construction policy, not the proposed Sci-VoI-RHI.",
        "Direct LLM judge and Sci-VoI-RHI trajectory replay are intentionally marked pending rather than assigned mock scores.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baselines on prompt-conditioned trajectories.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    report = evaluate_prompt_baselines(args.trajectories, args.out_dir, budget_fraction=args.budget_fraction, seed=args.seed)
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
