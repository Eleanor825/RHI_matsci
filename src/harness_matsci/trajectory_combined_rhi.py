from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .historical import grouped_four_way_split
from .rhi import train_rhi_from_splits
from .schema import ActionRecord


def _rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _action_records(rows: list[dict[str, Any]]) -> list[ActionRecord]:
    records = []
    for row in rows:
        record = ActionRecord.from_json(row["source_action_record"])
        records.append(record)
    return records


def _outcome_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outcomes = {}
    for row in rows:
        source = row["source_action_record"]
        outcomes[str(row["source_record_id"])] = {
            "executed_candidate": bool(row.get("executed_candidate", False)),
            "hidden_label": int(source.get("label", row.get("outcome", {}).get("label", 0))),
            "hidden_utility": float(source.get("utility", row.get("outcome", {}).get("utility", 0.0)) or 0.0),
            "cost": float(row.get("cost", 0.0) or 0.0),
            "steps": row.get("steps", []),
        }
    return outcomes


def _trajectory_metrics(
    records: list[ActionRecord],
    rows_by_id: dict[str, dict[str, Any]],
    gate: Any,
    *,
    selected_records: list[ActionRecord] | None = None,
) -> dict[str, Any]:
    decisions = gate.decisions(records)
    selected = selected_records if selected_records is not None else [
        record for record, decision in zip(records, decisions) if decision.selected
    ]
    successes = []
    realized_utilities = []
    net_utilities = []
    costs = []
    lengths = []
    terminal_actions: Counter[str] = Counter()
    for record in selected:
        outcome = rows_by_id[record.record_id]
        label = int(outcome["hidden_label"])
        cost = float(outcome["cost"])
        success = int(label == 1)
        utility = float(outcome["hidden_utility"]) if success else 0.0
        successes.append(success)
        realized_utilities.append(utility)
        net_utilities.append(utility - 0.15 * cost)
        costs.append(cost)
        steps = outcome.get("steps", [])
        lengths.append(len(steps))
        if steps:
            terminal_actions[str(steps[-1].get("agent_output", {}).get("action_id", "unknown"))] += 1
    n = len(records)
    selected_n = len(selected)
    return {
        "n_trajectories": n,
        "selected": selected_n,
        "coverage": selected_n / n if n else 0.0,
        "trajectory_success_rate": sum(successes) / selected_n if selected_n else 0.0,
        "trajectory_risk": 1.0 - (sum(successes) / selected_n if selected_n else 0.0),
        "mean_realized_utility_selected": sum(realized_utilities) / selected_n if selected_n else 0.0,
        "mean_net_utility_selected": sum(net_utilities) / selected_n if selected_n else 0.0,
        "net_utility_per_trajectory": sum(net_utilities) / n if n else 0.0,
        "mean_cost_selected": sum(costs) / selected_n if selected_n else 0.0,
        "mean_steps_selected": sum(lengths) / selected_n if selected_n else 0.0,
        "terminal_actions": dict(terminal_actions),
    }


def _top_budget_records(records: list[ActionRecord], gate: Any, fraction: float) -> list[ActionRecord]:
    probabilities = gate.predict_proba(records)
    n_selected = max(1, int(len(records) * fraction))
    order = sorted(range(len(records)), key=lambda index: (-probabilities[index], records[index].record_id))
    return [records[index] for index in order[:n_selected]]


def run_combined_feedback_rhi(
    trajectory_path: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 1729,
    iterations: int = 3,
    epochs: int = 45,
    budget_fraction: float = 0.10,
) -> dict[str, Any]:
    """Run RHI where action errors and complete-route outcomes co-train mutations."""
    rows = _rows(trajectory_path)
    records = _action_records(rows)
    rows_by_id = _outcome_map(rows)
    splits = grouped_four_way_split(records, seed=seed, train_fraction=0.60, feedback_fraction=0.15, acceptance_fraction=0.10)
    outcomes = _outcome_map(rows)
    report = train_rhi_from_splits(
        splits["train"],
        splits["feedback"],
        splits["test"],
        iterations=iterations,
        seed=seed,
        alpha=0.10,
        budget_fraction=budget_fraction,
        epochs=epochs,
        learning_rate=0.08,
        l2=0.01,
        acceptance_records=splits["acceptance"],
        acceptance_policy="guarded",
        trajectory_outcomes=outcomes,
    )
    checkpoint_metrics = []
    for checkpoint in report["active_checkpoints"]:
        from .training import TrainedGate
        gate = TrainedGate.from_json(checkpoint["gate"])
        budget_selected = _top_budget_records(splits["test"], gate, budget_fraction)
        checkpoint_metrics.append({
            "round": checkpoint["round"],
            "name": checkpoint["name"],
            "accepted_revision": checkpoint["accepted_revision"],
            "runtime_route": _trajectory_metrics(splits["test"], rows_by_id, gate),
            "top_budget": _trajectory_metrics(
                splits["test"], rows_by_id, gate, selected_records=budget_selected
            ),
        })
    result = {
        "schema_version": "combined-action-trajectory-feedback-rhi-v1",
        "trajectory_path": str(trajectory_path),
        "n_trajectories": len(rows),
        "seed": seed,
        "iterations": iterations,
        "epochs": epochs,
        "budget_fraction": budget_fraction,
        "feedback_schema": {
            "action_level": ["confidently_wrong", "overconfident", "calibration_error", "evidence_conflict", "high_ood", "over_abstention"],
            "trajectory_level": ["wrong_final_commit", "missed_good_opportunity", "failed_recovery", "costly_failure", "over_verification", "unnecessary_abstention"],
        },
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "rhi": report,
        "trajectory_test_checkpoints": checkpoint_metrics,
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trajectory_combined_rhi_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Combined Action + Trajectory Feedback RHI",
        "",
        "This experiment uses complete prompt-generated trajectories. Each action contributes action-level feedback; the full route contributes outcome feedback. One aggregate feedback object produces one candidate harness per RHI round. Individual actions do not independently mutate the harness.",
        "",
        f"- Trajectories: {len(rows)} (300 per task)",
        f"- Splits: {result['split_sizes']}",
        f"- Iterations: {iterations}",
        "- Test outcomes are used only after harness selection.",
        "",
        "## Test checkpoints",
        "",
        "| Round | Harness | Accepted | Coverage (top 10%) | Risk | Success | Net utility / trajectory | Mean steps |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for item in checkpoint_metrics:
        metrics = item["top_budget"]
        lines.append(
            f"| {item['round']} | {item['name']} | {item['accepted_revision']} | "
            f"{item['top_budget']['coverage']:.3f} | {item['top_budget']['trajectory_risk']:.3f} | "
            f"{item['top_budget']['trajectory_success_rate']:.3f} | "
            f"{item['top_budget']['net_utility_per_trajectory']:.4f} | "
            f"{item['top_budget']['mean_steps_selected']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Action-level feedback identifies local calibration, evidence, and uncertainty failures.",
        "- Trajectory-level feedback identifies wrong final commitments, failed recovery, unnecessary verification, and missed opportunities.",
        "- The proposer mutates the declarative harness once per round; acceptance compares the candidate against its predecessor on a held-out acceptance shard.",
        "- The current prompt backend is deterministic and offline, not an LLM-generated runtime trace; this is a reproducible trajectory benchmark rather than an online MatBot result.",
        "- Runtime-route metrics are also stored in JSON; the table uses a fixed top-10% budget so every checkpoint is compared at equal action budget.",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run combined action- and trajectory-feedback RHI.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()
    result = run_combined_feedback_rhi(args.trajectories, args.out_dir, seed=args.seed, iterations=args.iterations, epochs=args.epochs)
    print(json.dumps({"split_sizes": result["split_sizes"], "checkpoints": result["trajectory_test_checkpoints"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
