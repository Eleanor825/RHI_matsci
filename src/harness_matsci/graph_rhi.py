from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .conditional_action_rhi import (
    ALL_CANDIDATE_SIGNALS,
    _feature_names,
    _report,
    _score,
    _train_external_gate,
    _train_task_conditional_gate,
    add_policy_features,
    load_external_split_records,
)


INITIAL_SIGNALS = {"verbal_confidence", "cost", "reversibility"}

BRANCHES = {
    "H1_evidence_source": {
        "materials_pairwise_preference::evidence_support",
        "materials_pairwise_preference::source_reliability",
        "unique_materials_screening::evidence_support",
        "unique_materials_screening::source_reliability",
    },
    "H2_ood_stability": {
        "materials_pairwise_preference::perturbation_stability",
        "unique_materials_screening::ood_score",
        "extreme_property_discovery::ood_score",
        "extreme_property_discovery::perturbation_stability",
    },
}


def _fit_node(rows: dict[str, list[Any]], signals: set[str], *, alpha: float, budget_fraction: float, epochs: int, conditional: bool = True) -> tuple[Any, dict[str, Any]]:
    prepared = {split: add_policy_features(items, signals) for split, items in rows.items()}
    names = _feature_names(prepared["train"] + prepared["feedback"], signals)
    if conditional:
        gate = _train_task_conditional_gate(prepared["train"], prepared["feedback"], signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
    else:
        gate = _train_external_gate(prepared["train"], prepared["feedback"], names, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs)
    return gate, {"gate": gate, "signals": sorted(signals), "features": len(names), "prepared": prepared, "acceptance": _report(prepared["acceptance"], gate, budget_fraction), "test": _report(prepared["test"], gate, budget_fraction)}


def _accept(candidate: dict[str, Any], parent: dict[str, Any], *, alpha: float) -> bool:
    candidate_risk = float(candidate["acceptance"]["fixed_budget"]["risk"])
    parent_risk = float(parent["acceptance"]["fixed_budget"]["risk"])
    guard = candidate_risk <= parent_risk + 0.01
    if parent_risk <= alpha:
        guard = guard and candidate_risk <= alpha
    return bool(guard and _score(candidate["acceptance"]) > _score(parent["acceptance"]) + 1e-4)


def run_external_graph_rhi(
    roots: dict[str, str | Path],
    *,
    fold: int = 0,
    alpha: float = 0.10,
    budget_fraction: float = 0.10,
    epochs: int = 80,
) -> dict[str, Any]:
    rows = {split: [] for split in ("train", "feedback", "acceptance", "test")}
    task_counts = {}
    for task_name, root in roots.items():
        loaded = load_external_split_records(root, fold=fold)
        task_counts[task_name] = {split: len(items) for split, items in loaded.items()}
        for split in rows:
            rows[split].extend(loaded[split])

    nodes: dict[str, dict[str, Any]] = {}
    h0_gate, h0 = _fit_node(rows, INITIAL_SIGNALS, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs, conditional=False)
    nodes["H0"] = {key: value for key, value in h0.items() if key not in {"gate", "prepared"}}
    branches: dict[str, dict[str, Any]] = {}
    for node_id, additions in BRANCHES.items():
        signals = INITIAL_SIGNALS | additions
        _, node = _fit_node(rows, signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs, conditional=True)
        accepted = _accept(node, h0, alpha=alpha)
        branches[node_id] = {key: value for key, value in node.items() if key not in {"gate", "prepared"}}
        branches[node_id].update({"parent_ids": ["H0"], "mutation": sorted(additions), "accepted": accepted})
        nodes[node_id] = branches[node_id]

    accepted_branches = [node_id for node_id, node in branches.items() if node["accepted"]]
    merge_signals = set(INITIAL_SIGNALS)
    for node_id in accepted_branches:
        merge_signals.update(BRANCHES[node_id])
    _, merged = _fit_node(rows, merge_signals, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs, conditional=True)
    merge_parent = {"acceptance": h0["acceptance"]}
    merge_accepted = len(accepted_branches) >= 2 and _accept(merged, merge_parent, alpha=alpha)
    merge_node = {key: value for key, value in merged.items() if key not in {"gate", "prepared"}}
    merge_node.update({"parent_ids": accepted_branches, "mutation": "merge", "accepted": merge_accepted})
    nodes["H3_merge"] = merge_node

    active_id = "H3_merge" if merge_accepted else (accepted_branches[0] if accepted_branches else "H0")
    return {
        "method": "Graph-RHI",
        "fold": fold,
        "task_counts": task_counts,
        "split_sizes": {split: len(items) for split, items in rows.items()},
        "nodes": nodes,
        "edges": [
            {"parent": "H0", "child": node_id, "type": "mutation"} for node_id in BRANCHES
        ] + ([{"parent": node_id, "child": "H3_merge", "type": "merge"} for node_id in accepted_branches]),
        "accepted_branches": accepted_branches,
        "merge_accepted": merge_accepted,
        "active_node": active_id,
        "final_test": nodes[active_id]["test"],
        "branch_gain": {node_id: _score(node["acceptance"]) - _score(nodes["H0"]["acceptance"]) for node_id, node in branches.items()},
        "merge_gain": _score(merge_node["acceptance"]) - _score(nodes["H0"]["acceptance"]),
    }


def run_external_graph_suite(
    roots: dict[str, str | Path],
    *,
    folds: int = 5,
    alpha: float = 0.10,
    budget_fraction: float = 0.10,
    epochs: int = 80,
) -> dict[str, Any]:
    results = [run_external_graph_rhi(roots, fold=fold, alpha=alpha, budget_fraction=budget_fraction, epochs=epochs) for fold in range(folds)]

    def mean(node: str, key: str) -> float:
        return sum(float(result["nodes"][node]["test"]["fixed_budget"][key]) for result in results) / len(results)

    aggregate = {}
    for node in ("H0", "H1_evidence_source", "H2_ood_stability", "H3_merge"):
        for key in ("risk", "hit_rate", "mean_utility", "coverage"):
            aggregate[f"{node}_{key}_mean"] = mean(node, key)
    for node in ("H1_evidence_source", "H2_ood_stability"):
        aggregate[f"{node}_acceptance_rate"] = sum(bool(result["nodes"][node]["accepted"]) for result in results) / len(results)
    aggregate["any_branch_acceptance_rate"] = sum(len(result["accepted_branches"]) > 0 for result in results) / len(results)
    aggregate["merge_acceptance_rate"] = sum(result["merge_accepted"] for result in results) / len(results)
    aggregate["active_merge_rate"] = sum(result["active_node"] == "H3_merge" for result in results) / len(results)
    return {"method": "Graph-RHI", "folds": list(range(folds)), "results": results, "aggregate": aggregate}


def save_graph_suite(report: dict[str, Any], out_dir: str | Path) -> None:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aggregate = report["aggregate"]
    lines = [
        "# Graph-RHI: branching and merge experiment",
        "",
        "All nodes use the same five folds and fixed top-10% action budget.",
        "H0 is fixed; H1/H2 are independent mutation branches; H3 merges accepted branches.",
        "",
        "| Method | Risk ↓ | Hit rate ↑ | Mean utility ↑ | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for node, label in (("H0", "H0 fixed"), ("H1_evidence_source", "H1 evidence/source"), ("H2_ood_stability", "H2 OOD/stability"), ("H3_merge", "H3 merged")):
        lines.append(f"| {label} | {aggregate[node+'_risk_mean']:.4f} | {aggregate[node+'_hit_rate_mean']:.4f} | {aggregate[node+'_mean_utility_mean']:.4f} | {aggregate[node+'_coverage_mean']:.4f} |")
    lines += ["", f"H1 acceptance rate: `{aggregate['H1_evidence_source_acceptance_rate']:.2f}`.", f"H2 acceptance rate: `{aggregate['H2_ood_stability_acceptance_rate']:.2f}`.", f"Any-branch acceptance rate: `{aggregate['any_branch_acceptance_rate']:.2f}`.", f"True merge acceptance rate: `{aggregate['merge_acceptance_rate']:.2f}`.", f"Active merged node rate: `{aggregate['active_merge_rate']:.2f}`.", "", "The graph retains branch lineage and only permits a true merge when at least two independently accepted branches are available."]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
