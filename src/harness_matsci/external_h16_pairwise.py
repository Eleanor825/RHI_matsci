from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .external_h14 import balanced_group_folds, weighted_group_partition
from .external_h15_unique import (
    UniqueMaterial,
    load_external_h15_unique_materials,
)
from .schema import ActionRecord


TASK = "external_pairwise_log_kvrh"


@dataclass(frozen=True)
class PairwiseFold:
    fold: int
    records_by_split: dict[str, list[ActionRecord]]
    groups_by_split: dict[str, tuple[str, ...]]


def build_external_h16_pairwise_fold(
    source_path: str | Path,
    *,
    fold: int,
    n_folds: int = 5,
    seed: int = 1601,
    pair_multiplier: float = 1.0,
) -> PairwiseFold:
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
    records = _build_records(
        splits,
        seed=seed + fold,
        pair_multiplier=pair_multiplier,
    )
    _validate(records, groups)
    return PairwiseFold(fold, records, groups)


def write_external_h16_pairwise_fold(
    fold: PairwiseFold, output_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root) / f"fold_{fold.fold}"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, records in fold.records_by_split.items():
        (root / f"{TASK}.{split}.jsonl").write_text(
            "\n".join(json.dumps(record.to_json(), sort_keys=True) for record in records)
            + "\n",
            encoding="utf-8",
        )
        counts[split] = len(records)
    manifest = {
        "fold": fold.fold,
        "task": TASK,
        "counts": counts,
        "groups_by_split": {
            split: list(groups) for split, groups in fold.groups_by_split.items()
        },
        "material_group_disjoint": True,
        "pair_generation": "train-surrogate-neighbor pairing within each material split",
        "hidden_outcome_contract": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _build_records(splits, *, seed, pair_multiplier):
    train_targets = sorted(material.target for material in splits["train"])
    ensemble = _fit_ensemble(splits["train"], seed=seed)
    output = {}
    for split_index, (split, materials) in enumerate(splits.items()):
        predictions = _predict_ensemble(ensemble, materials)
        desired = max(1, int(round(len(materials) * pair_multiplier)))
        output[split] = _make_pairs(
            materials,
            predictions,
            train_targets,
            desired=desired,
            seed=seed + split_index * 503,
            split=split,
        )
    return output


def _make_pairs(materials, predictions, train_targets, *, desired, seed, split):
    rng = random.Random(seed)
    order = sorted(range(len(materials)), key=lambda index: predictions[index][0])
    position = {index: rank for rank, index in enumerate(order)}
    seen = set()
    rows = []
    attempts = 0
    while len(rows) < desired and attempts < desired * 50:
        attempts += 1
        left_index = rng.randrange(len(materials))
        left_rank = position[left_index]
        radius = rng.randint(1, min(40, len(materials) - 1))
        direction = -1 if rng.random() < 0.5 else 1
        right_rank = max(0, min(len(materials) - 1, left_rank + direction * radius))
        right_index = order[right_rank]
        if right_index == left_index:
            continue
        key = tuple(sorted((left_index, right_index)))
        if key in seen:
            continue
        seen.add(key)
        left = materials[left_index]
        right = materials[right_index]
        left_prediction, left_variance = predictions[left_index]
        right_prediction, right_variance = predictions[right_index]
        if left_prediction >= right_prediction:
            chosen, alternative = left, right
            chosen_prediction, alternative_prediction = left_prediction, right_prediction
        else:
            chosen, alternative = right, left
            chosen_prediction, alternative_prediction = right_prediction, left_prediction
        chosen_percentile = _empirical_percentile(train_targets, chosen.target)
        alternative_percentile = _empirical_percentile(train_targets, alternative.target)
        utility = max(0.0, chosen_percentile - alternative_percentile)
        margin = chosen_prediction - alternative_prediction
        uncertainty = math.sqrt(max(0.0, left_variance + right_variance))
        rows.append(
            ActionRecord(
                record_id=(
                    f"external-log-kvrh-pair::{split}::{chosen.mbid}::{alternative.mbid}"
                ),
                benchmark=TASK,
                split=split,
                visible_context=(
                    "Choose the material expected to have higher bulk modulus. "
                    f"Candidate A: {chosen.composition}, {chosen.crystal_system}, "
                    f"space group {chosen.space_group}. Candidate B: "
                    f"{alternative.composition}, {alternative.crystal_system}, "
                    f"space group {alternative.space_group}."
                ),
                candidate_action="Choose candidate A for experimental follow-up.",
                action_type="choose_candidate",
                evidence=[
                    f"train-only surrogate pair margin={margin:.6f}",
                    f"ensemble disagreement={uncertainty:.6f}",
                ],
                features={
                    "verbal_confidence": 1.0 / (1.0 + math.exp(-margin)),
                    "evidence_support": min(1.0, abs(margin)),
                    "evidence_conflict": max(0.0, 1.0 - abs(margin)),
                    "perturbation_stability": max(0.0, 1.0 - uncertainty),
                    "ood_score": min(1.0, uncertainty),
                    "source_reliability": 0.88,
                    "cost": 0.35,
                    "reversibility": 0.80,
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
                },
            )
        )
    if len(rows) != desired:
        raise RuntimeError(f"constructed {len(rows)} of {desired} requested pairs")
    return rows


def _fit_ensemble(materials: list[UniqueMaterial], *, seed: int):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    matrix = np.asarray([material.vector for material in materials], dtype=float)
    targets = np.asarray([material.target for material in materials], dtype=float)
    return tuple(
        ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=5,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.85,
            random_state=seed * 1009 + replica * 97,
            n_jobs=-1,
        ).fit(matrix, targets)
        for replica in range(3)
    )


def _predict_ensemble(models, materials):
    import numpy as np

    matrix = np.asarray([material.vector for material in materials], dtype=float)
    predictions = np.asarray([model.predict(matrix) for model in models], dtype=float)
    return list(zip(np.mean(predictions, axis=0), np.var(predictions, axis=0)))


def _empirical_percentile(values, value):
    import bisect

    ordered = values if values == sorted(values) else sorted(values)
    return (bisect.bisect_right(ordered, value) + 0.5) / (len(ordered) + 1.0)


def _validate(records, groups):
    sets = {split: set(values) for split, values in groups.items()}
    names = tuple(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError(f"group leakage between {left} and {right}")
    for split, rows in records.items():
        for row in rows:
            if row.metadata["chosen_group_id"] not in sets[split]:
                raise ValueError(f"chosen material leakage in {split}")
            if row.metadata["alternative_group_id"] not in sets[split]:
                raise ValueError(f"alternative material leakage in {split}")
