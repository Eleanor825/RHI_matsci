from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .external_h14 import balanced_group_folds, weighted_group_partition
from .external_h15_unique import load_external_h15_unique_materials
from .external_h16_pairwise import _empirical_percentile, _fit_ensemble, _predict_ensemble
from .external_h16_pairwise_tools import _fit_regressor, _predict_many
from .schema import ActionRecord


TASK = "external_pairwise_mixed_log_kvrh"
REGIME_PROPORTIONS = {"easy": 0.40, "ambiguous": 0.30, "ood": 0.30}


@dataclass(frozen=True)
class PairwiseMixedFold:
    fold: int
    records_by_split: dict[str, list[ActionRecord]]
    groups_by_split: dict[str, tuple[str, ...]]


def build_external_h17_pairwise_fold(
    source_path: str | Path,
    *,
    fold: int,
    n_folds: int = 5,
    seed: int = 1701,
    pair_multiplier: float = 1.0,
) -> PairwiseMixedFold:
    materials = load_external_h15_unique_materials(source_path)
    group_sizes: dict[str, int] = {}
    for material in materials:
        group_sizes[material.group_id] = group_sizes.get(material.group_id, 0) + 1
    outer = balanced_group_folds(group_sizes, n_folds=n_folds, seed=seed)
    test_groups = set(outer[fold])
    remaining = {
        group: size for group, size in group_sizes.items() if group not in test_groups
    }
    development = weighted_group_partition(
        remaining,
        weights={"train": 0.50, "feedback": 0.15, "acceptance": 0.20},
        seed=seed + fold * 1009,
    )
    groups = {
        **{split: tuple(sorted(values)) for split, values in development.items()},
        "test": tuple(sorted(test_groups)),
    }
    splits = {
        split: [material for material in materials if material.group_id in set(group_ids)]
        for split, group_ids in groups.items()
    }
    records = _build_records(splits, seed=seed + fold, pair_multiplier=pair_multiplier)
    _validate(records, groups)
    return PairwiseMixedFold(fold, records, groups)


def write_external_h17_pairwise_fold(
    fold: PairwiseMixedFold, output_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root) / f"fold_{fold.fold}"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    regime_counts = {}
    for split, records in fold.records_by_split.items():
        (root / f"{TASK}.{split}.jsonl").write_text(
            "\n".join(json.dumps(record.to_json(), sort_keys=True) for record in records)
            + "\n",
            encoding="utf-8",
        )
        counts[split] = len(records)
        regime_counts[split] = {
            regime: sum(record.metadata["pair_regime"] == regime for record in records)
            for regime in REGIME_PROPORTIONS
        }
    manifest = {
        "fold": fold.fold,
        "task": TASK,
        "counts": counts,
        "regime_counts": regime_counts,
        "regime_proportions": REGIME_PROPORTIONS,
        "groups_by_split": {
            split: list(groups) for split, groups in fold.groups_by_split.items()
        },
        "material_group_disjoint": True,
        "pair_generation": (
            "train-only surrogate stratification into easy, ambiguous, and OOD regimes"
        ),
        "stratification_uses_hidden_outcomes": False,
        "hidden_outcome_contract": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _build_records(splits, *, seed, pair_multiplier):
    train = splits["train"]
    train_targets = tuple(sorted(material.target for material in train))
    ensemble = _fit_ensemble(train, seed=seed)
    composition = _fit_regressor(train, source="composition", seed=seed + 97)
    structure = _fit_regressor(train, source="structure", seed=seed + 193)
    output = {}
    for split_index, (split, materials) in enumerate(splits.items()):
        desired = max(1, int(round(len(materials) * pair_multiplier)))
        output[split] = _make_mixed_pairs(
            materials,
            combined_predictions=_predict_ensemble(ensemble, materials),
            composition_predictions=_predict_many(composition, materials),
            structure_predictions=_predict_many(structure, materials),
            train_targets=train_targets,
            desired=desired,
            seed=seed + split_index * 503,
            split=split,
        )
    return output


def _make_mixed_pairs(
    materials,
    *,
    combined_predictions,
    composition_predictions,
    structure_predictions,
    train_targets,
    desired,
    seed,
    split,
):
    rng = random.Random(seed)
    candidates = []
    seen = set()
    attempts = 0
    pool_target = desired * 60
    while len(candidates) < pool_target and attempts < pool_target * 4:
        attempts += 1
        left_index = rng.randrange(len(materials))
        right_index = rng.randrange(len(materials))
        if left_index == right_index:
            continue
        key = tuple(sorted((left_index, right_index)))
        if key in seen:
            continue
        seen.add(key)
        if combined_predictions[left_index][0] >= combined_predictions[right_index][0]:
            chosen_index, alternative_index = left_index, right_index
        else:
            chosen_index, alternative_index = right_index, left_index
        chosen_prediction, chosen_variance = combined_predictions[chosen_index]
        alternative_prediction, alternative_variance = combined_predictions[alternative_index]
        combined_margin = chosen_prediction - alternative_prediction
        uncertainty = math.sqrt(max(0.0, chosen_variance + alternative_variance))
        composition_margin = (
            composition_predictions[chosen_index]
            - composition_predictions[alternative_index]
        )
        structure_margin = (
            structure_predictions[chosen_index]
            - structure_predictions[alternative_index]
        )
        candidates.append(
            {
                "chosen_index": chosen_index,
                "alternative_index": alternative_index,
                "combined_margin": combined_margin,
                "uncertainty": uncertainty,
                "composition_margin": composition_margin,
                "structure_margin": structure_margin,
                "source_range": abs(composition_margin - structure_margin),
                "source_disagreement": float(composition_margin * structure_margin < 0.0),
                "chosen_prediction": chosen_prediction,
                "alternative_prediction": alternative_prediction,
            }
        )
    if len(candidates) < desired:
        raise RuntimeError("insufficient candidate pairs for mixed-regime construction")
    margin_q50 = _quantile([row["combined_margin"] for row in candidates], 0.50)
    margin_q75 = _quantile([row["combined_margin"] for row in candidates], 0.75)
    uncertainty_q50 = _quantile([row["uncertainty"] for row in candidates], 0.50)
    uncertainty_q75 = _quantile([row["uncertainty"] for row in candidates], 0.75)
    source_range_q75 = _quantile([row["source_range"] for row in candidates], 0.75)
    pools = {
        "easy": [
            row
            for row in candidates
            if row["combined_margin"] >= margin_q75
            and row["uncertainty"] <= uncertainty_q50
            and row["composition_margin"] > 0.0
            and row["structure_margin"] > 0.0
        ],
        "ambiguous": [
            row
            for row in candidates
            if row["combined_margin"] <= margin_q50
            and (
                row["source_disagreement"] > 0.0
                or row["source_range"] >= source_range_q75
            )
        ],
        "ood": [
            row
            for row in candidates
            if row["uncertainty"] >= uncertainty_q75
            and row["combined_margin"] <= margin_q75
        ],
    }
    scores = {
        "easy": lambda row: row["combined_margin"] / (row["uncertainty"] + 1e-6),
        "ambiguous": lambda row: (
            2.0 * row["source_disagreement"]
            + row["source_range"] / (source_range_q75 + 1e-6)
            - row["combined_margin"] / (margin_q50 + 1e-6)
        ),
        "ood": lambda row: (
            row["uncertainty"] / (uncertainty_q75 + 1e-6)
            - row["combined_margin"] / (margin_q75 + 1e-6)
        ),
    }
    quotas = _regime_quotas(desired)
    selected = []
    used = set()
    for regime in ("easy", "ambiguous", "ood"):
        ordered = sorted(
            pools[regime],
            key=lambda row: (
                -scores[regime](row),
                row["chosen_index"],
                row["alternative_index"],
            ),
        )
        for row in ordered:
            key = tuple(sorted((row["chosen_index"], row["alternative_index"])))
            if key in used:
                continue
            used.add(key)
            selected.append((regime, row))
            if sum(name == regime for name, _ in selected) == quotas[regime]:
                break
    if len(selected) != desired:
        raise RuntimeError(
            f"constructed {len(selected)} of {desired} requested stratified pairs"
        )
    return [
        _record_from_candidate(
            materials,
            row,
            regime=regime,
            split=split,
            train_targets=train_targets,
        )
        for regime, row in selected
    ]


def _record_from_candidate(materials, row, *, regime, split, train_targets):
    chosen = materials[row["chosen_index"]]
    alternative = materials[row["alternative_index"]]
    chosen_percentile = _empirical_percentile(train_targets, chosen.target)
    alternative_percentile = _empirical_percentile(train_targets, alternative.target)
    utility = max(0.0, chosen_percentile - alternative_percentile)
    estimated_utility = _pair_utility(
        train_targets, row["chosen_prediction"], row["alternative_prediction"]
    )
    uncertainty = row["uncertainty"]
    utility_low = _pair_utility(
        train_targets,
        row["chosen_prediction"] - uncertainty,
        row["alternative_prediction"] + uncertainty,
    )
    utility_high = _pair_utility(
        train_targets,
        row["chosen_prediction"] + uncertainty,
        row["alternative_prediction"] - uncertainty,
    )
    confidence = _logistic(row["combined_margin"] / (uncertainty + 1e-3))
    confidence_low = _logistic(
        (row["combined_margin"] - uncertainty) / (uncertainty + 1e-3)
    )
    confidence_high = _logistic(
        (row["combined_margin"] + uncertainty) / (uncertainty + 1e-3)
    )
    return ActionRecord(
        record_id=f"external-mixed-pair::{split}::{regime}::{chosen.mbid}::{alternative.mbid}",
        benchmark=TASK,
        split=split,
        visible_context=(
            "Choose the material expected to have higher bulk modulus. "
            f"Candidate A: {chosen.composition}, {chosen.crystal_system}, space group "
            f"{chosen.space_group}. Candidate B: {alternative.composition}, "
            f"{alternative.crystal_system}, space group {alternative.space_group}."
        ),
        candidate_action="Choose candidate A for experimental follow-up.",
        action_type="choose_candidate",
        evidence=[
            f"train-only surrogate pair margin={row['combined_margin']:.6f}",
            f"ensemble disagreement={uncertainty:.6f}",
            f"pre-execution regime={regime}",
        ],
        features={
            "verbal_confidence": confidence,
            "evidence_support": min(1.0, abs(row["combined_margin"])),
            "evidence_conflict": max(0.0, 1.0 - abs(row["combined_margin"])),
            "perturbation_stability": max(0.0, 1.0 - uncertainty),
            "ood_score": min(1.0, uncertainty),
            "source_reliability": 0.88,
            "cost": 0.35,
            "reversibility": 0.80,
            "agent_prior_margin": row["combined_margin"],
            "agent_prior_uncertainty": uncertainty,
            "agent_prior_confidence": confidence,
            "agent_prior_confidence_low": confidence_low,
            "agent_prior_confidence_high": confidence_high,
            "agent_prior_estimated_utility": estimated_utility,
            "agent_prior_estimated_utility_low": min(estimated_utility, utility_low),
            "agent_prior_estimated_utility_high": max(estimated_utility, utility_high),
        },
        label=int(chosen.target > alternative.target),
        utility=utility,
        metadata={
            "group_id": f"{chosen.group_id}::{alternative.group_id}",
            "chosen_group_id": chosen.group_id,
            "alternative_group_id": alternative.group_id,
            "source_dataset": "matbench_log_kvrh",
            "chosen_row_id": chosen.row_id,
            "alternative_row_id": alternative.row_id,
            "pair_regime": regime,
            "regime_assignment_uses_hidden_outcome": False,
        },
    )


def _regime_quotas(desired):
    easy = int(round(desired * REGIME_PROPORTIONS["easy"]))
    ambiguous = int(round(desired * REGIME_PROPORTIONS["ambiguous"]))
    return {"easy": easy, "ambiguous": ambiguous, "ood": desired - easy - ambiguous}


def _pair_utility(train_targets, chosen, alternative):
    return max(
        0.0,
        _empirical_percentile(train_targets, chosen)
        - _empirical_percentile(train_targets, alternative),
    )


def _quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _logistic(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate(records, groups):
    sets = {split: set(values) for split, values in groups.items()}
    names = tuple(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError(f"group leakage between {left} and {right}")
    for split, rows in records.items():
        if set(record.metadata["pair_regime"] for record in rows) != set(REGIME_PROPORTIONS):
            raise ValueError(f"missing pair regime in {split}")
        for row in rows:
            if row.metadata["chosen_group_id"] not in sets[split]:
                raise ValueError(f"chosen material leakage in {split}")
            if row.metadata["alternative_group_id"] not in sets[split]:
                raise ValueError(f"alternative material leakage in {split}")
