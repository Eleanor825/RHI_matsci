from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import ActionRecord


EXTERNAL_H14_TASKS = (
    "external_pairwise_dielectric",
    "external_unique_jdft2d",
    "external_extreme_phonons",
)


@dataclass(frozen=True)
class ExternalMaterial:
    dataset: str
    row_id: int
    structure: Any
    target: float
    group_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class ExternalFold:
    fold: int
    records_by_task_and_split: dict[str, dict[str, list[ActionRecord]]]
    group_ids_by_split: dict[str, tuple[str, ...]]


def build_external_h14_fold(
    snapshot_root: str | Path,
    *,
    fold: int,
    n_folds: int = 5,
    seed: int = 1401,
) -> ExternalFold:
    if not 0 <= fold < n_folds:
        raise ValueError("fold must lie in [0, n_folds)")
    datasets = _load_snapshots(snapshot_root)
    all_materials = [material for rows in datasets.values() for material in rows]
    group_sizes: dict[str, int] = {}
    for material in all_materials:
        group_sizes[material.group_id] = group_sizes.get(material.group_id, 0) + 1
    outer_folds = balanced_group_folds(group_sizes, n_folds=n_folds, seed=seed)
    test_groups = set(outer_folds[fold])
    remaining_sizes = {
        group: count for group, count in group_sizes.items() if group not in test_groups
    }
    development_groups = weighted_group_partition(
        remaining_sizes,
        weights={"train": 0.50, "feedback": 0.15, "acceptance": 0.20},
        seed=seed + fold * 1009,
    )
    group_ids_by_split = {
        **{split: tuple(sorted(groups)) for split, groups in development_groups.items()},
        "test": tuple(sorted(test_groups)),
    }
    material_splits = {
        dataset: {
            split: [
                material
                for material in materials
                if material.group_id in set(group_ids_by_split[split])
            ]
            for split in ("train", "feedback", "acceptance", "test")
        }
        for dataset, materials in datasets.items()
    }
    records = {
        "external_pairwise_dielectric": _build_pairwise_records(
            material_splits["matbench_dielectric"], seed=seed + fold
        ),
        "external_unique_jdft2d": _build_unique_records(
            material_splits["matbench_jdft2d"], seed=seed + fold
        ),
        "external_extreme_phonons": _build_extreme_records(
            material_splits["matbench_phonons"], seed=seed + fold
        ),
    }
    _validate_fold(records, group_ids_by_split)
    return ExternalFold(fold, records, group_ids_by_split)


def write_external_h14_fold(fold: ExternalFold, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root) / f"fold_{fold.fold}"
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    counts = {}
    for task, split_records in fold.records_by_task_and_split.items():
        counts[task] = {}
        for split, records in split_records.items():
            path = root / f"{task}.{split}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            files[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "records": len(records),
            }
            counts[task][split] = len(records)
    manifest = {
        "benchmark": "external_h14",
        "fold": fold.fold,
        "tasks": list(EXTERNAL_H14_TASKS),
        "counts": counts,
        "groups_by_split": {
            split: list(groups) for split, groups in fold.group_ids_by_split.items()
        },
        "files": files,
        "leakage_contract": {
            "group_key": "reduced_chemical_system",
            "group_disjoint": True,
            "train_only_surrogates": True,
            "train_only_utility_normalization": True,
            "train_only_novelty_reference": True,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def balanced_group_folds(
    group_sizes: dict[str, int], *, n_folds: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    if n_folds < 2:
        raise ValueError("at least two folds are required")
    if len(group_sizes) < n_folds:
        raise ValueError("group count must be at least the number of folds")
    loads = [0] * n_folds
    groups = [[] for _ in range(n_folds)]
    ordered = sorted(
        group_sizes,
        key=lambda group: (-group_sizes[group], _stable_hash(seed, group)),
    )
    for group in ordered:
        fold = min(
            range(n_folds),
            key=lambda index: (loads[index], _stable_hash(seed, group, str(index))),
        )
        groups[fold].append(group)
        loads[fold] += group_sizes[group]
    return tuple(tuple(sorted(fold_groups)) for fold_groups in groups)


def weighted_group_partition(
    group_sizes: dict[str, int], *, weights: dict[str, float], seed: int
) -> dict[str, tuple[str, ...]]:
    if not group_sizes:
        raise ValueError("cannot partition empty groups")
    if any(weight <= 0.0 for weight in weights.values()):
        raise ValueError("partition weights must be positive")
    total_weight = sum(weights.values())
    normalized = {name: weight / total_weight for name, weight in weights.items()}
    target = {
        name: normalized[name] * sum(group_sizes.values()) for name in normalized
    }
    loads = {name: 0 for name in normalized}
    output = {name: [] for name in normalized}
    ordered = sorted(
        group_sizes,
        key=lambda group: (-group_sizes[group], _stable_hash(seed, group)),
    )
    for group in ordered:
        split = min(
            normalized,
            key=lambda name: (
                loads[name] / max(1e-12, target[name]),
                _stable_hash(seed, group, name),
            ),
        )
        output[split].append(group)
        loads[split] += group_sizes[group]
    if any(not groups for groups in output.values()):
        raise ValueError("weighted partition produced an empty split")
    return {name: tuple(sorted(groups)) for name, groups in output.items()}


def _load_snapshots(snapshot_root: str | Path) -> dict[str, list[ExternalMaterial]]:
    from matminer.utils.io import load_dataframe_from_json

    root = Path(snapshot_root)
    specifications = {
        "matbench_dielectric": ("matbench_dielectric.json.gz", "n"),
        "matbench_jdft2d": ("matbench_jdft2d.json.gz", "exfoliation_en"),
        "matbench_phonons": ("matbench_phonons.json.gz", "last phdos peak"),
    }
    output = {}
    for dataset, (filename, target_column) in specifications.items():
        dataframe = load_dataframe_from_json(root / filename, pbar=False, decode=True)
        materials = []
        for row_id, row in dataframe.iterrows():
            structure = row["structure"]
            materials.append(
                ExternalMaterial(
                    dataset=dataset,
                    row_id=int(row_id),
                    structure=structure,
                    target=float(row[target_column]),
                    group_id=str(structure.composition.chemical_system),
                    vector=tuple(_structure_vector(structure)),
                )
            )
        output[dataset] = materials
    return output


def _build_pairwise_records(
    splits: dict[str, list[ExternalMaterial]], *, seed: int
) -> dict[str, list[ActionRecord]]:
    model = _fit_property_ensemble(splits["train"], seed=seed)
    train_targets = sorted(material.target for material in splits["train"])
    output = {}
    for split, materials in splits.items():
        predictions = _predict_ensemble(model, materials)
        ordered = sorted(
            range(len(materials)),
            key=lambda index: (
                predictions[index][0],
                _stable_hash(seed, split, str(materials[index].row_id)),
            ),
        )
        records = []
        for pair_index in range(0, len(ordered) - 1, 2):
            left = materials[ordered[pair_index]]
            right = materials[ordered[pair_index + 1]]
            left_prediction = predictions[ordered[pair_index]]
            right_prediction = predictions[ordered[pair_index + 1]]
            choose_left = left_prediction[0] >= right_prediction[0]
            chosen = left if choose_left else right
            alternative = right if choose_left else left
            chosen_prediction = left_prediction if choose_left else right_prediction
            alternative_prediction = right_prediction if choose_left else left_prediction
            label = int(chosen.target > alternative.target)
            chosen_percentile = _empirical_percentile(train_targets, chosen.target)
            alternative_percentile = _empirical_percentile(train_targets, alternative.target)
            utility = max(0.0, chosen_percentile - alternative_percentile)
            margin = chosen_prediction[0] - alternative_prediction[0]
            uncertainty = math.sqrt(chosen_prediction[1] + alternative_prediction[1])
            records.append(
                ActionRecord(
                    record_id=(
                        f"external-dielectric-pair::{split}::{chosen.row_id}::"
                        f"{alternative.row_id}"
                    ),
                    benchmark="external_pairwise_dielectric",
                    split=split,
                    visible_context=(
                        "Choose the material expected to have the higher dielectric target. "
                        f"Candidate A: {_structure_summary(chosen.structure)}. "
                        f"Candidate B: {_structure_summary(alternative.structure)}."
                    ),
                    candidate_action="Choose candidate A for experimental follow-up.",
                    action_type="choose_candidate",
                    evidence=[
                        f"train-only surrogate margin={margin:.6f}",
                        f"ensemble disagreement={uncertainty:.6f}",
                    ],
                    features=_common_features(
                        margin=margin,
                        uncertainty=uncertainty,
                        cost=0.35,
                        reversibility=0.80,
                    ),
                    label=label,
                    utility=utility,
                    metadata={
                        "group_id": (
                            f"external_pair::{chosen.group_id}::{alternative.group_id}"
                        ),
                        "source_dataset": "matbench_dielectric",
                        "chosen_row_id": chosen.row_id,
                        "alternative_row_id": alternative.row_id,
                    },
                )
            )
        output[split] = records
    return output


def _build_unique_records(
    splits: dict[str, list[ExternalMaterial]], *, seed: int
) -> dict[str, list[ActionRecord]]:
    model = _fit_property_ensemble(splits["train"], seed=seed + 101)
    train_targets = sorted(material.target for material in splits["train"])
    novelty_reference = _NoveltyReference.fit(splits["train"])
    train_scores = [
        0.5 * (1.0 - _empirical_percentile(train_targets, material.target))
        + 0.5 * novelty_reference.percentile(material, exclude_self=True)
        for material in splits["train"]
    ]
    threshold = _quantile(train_scores, 0.90)
    output = {}
    for split, materials in splits.items():
        predictions = _predict_ensemble(model, materials)
        records = []
        for material, (prediction, variance) in zip(materials, predictions):
            performance = 1.0 - _empirical_percentile(train_targets, material.target)
            novelty = novelty_reference.percentile(
                material, exclude_self=split == "train"
            )
            utility = 0.5 * (performance + novelty)
            predicted_performance = 1.0 - _empirical_percentile(
                train_targets, prediction
            )
            low_fidelity_novelty = _coarse_novelty(material, splits["train"])
            surrogate_score = 0.5 * (predicted_performance + low_fidelity_novelty)
            margin = surrogate_score - threshold
            uncertainty = math.sqrt(max(0.0, variance))
            records.append(
                ActionRecord(
                    record_id=f"external-jdft2d-unique::{material.row_id}",
                    benchmark="external_unique_jdft2d",
                    split=split,
                    visible_context=(
                        "Screen a 2D material for low exfoliation energy and structural "
                        f"uniqueness. Candidate: {_structure_summary(material.structure)}."
                    ),
                    candidate_action="Advance this 2D material for unique-discovery validation.",
                    action_type="choose_candidate",
                    evidence=[
                        f"train-only low-fidelity discovery margin={margin:.6f}",
                        f"property ensemble disagreement={uncertainty:.6f}",
                    ],
                    features=_common_features(
                        margin=margin,
                        uncertainty=uncertainty,
                        cost=0.50,
                        reversibility=0.60,
                    ),
                    label=int(utility >= threshold),
                    utility=utility,
                    metadata={
                        "group_id": material.group_id,
                        "source_dataset": "matbench_jdft2d",
                        "source_row_id": material.row_id,
                    },
                )
            )
        output[split] = records
    return output


def _build_extreme_records(
    splits: dict[str, list[ExternalMaterial]], *, seed: int
) -> dict[str, list[ActionRecord]]:
    model = _fit_property_ensemble(splits["train"], seed=seed + 211)
    train_targets = sorted(material.target for material in splits["train"])
    threshold = _quantile(train_targets, 0.90)
    output = {}
    for split, materials in splits.items():
        predictions = _predict_ensemble(model, materials)
        records = []
        for material, (prediction, variance) in zip(materials, predictions):
            utility = _empirical_percentile(train_targets, material.target)
            margin = prediction - threshold
            uncertainty = math.sqrt(max(0.0, variance))
            records.append(
                ActionRecord(
                    record_id=f"external-phonon-extreme::{material.row_id}",
                    benchmark="external_extreme_phonons",
                    split=split,
                    visible_context=(
                        "Find a material in the high extreme of the phonon density-of-states "
                        f"peak. Candidate: {_structure_summary(material.structure)}."
                    ),
                    candidate_action="Advance this material for extreme-phonon validation.",
                    action_type="choose_candidate",
                    evidence=[
                        f"train-only surrogate extreme margin={margin:.6f}",
                        f"property ensemble disagreement={uncertainty:.6f}",
                    ],
                    features=_common_features(
                        margin=margin,
                        uncertainty=uncertainty,
                        cost=0.60,
                        reversibility=0.50,
                    ),
                    label=int(material.target >= threshold),
                    utility=utility,
                    metadata={
                        "group_id": material.group_id,
                        "source_dataset": "matbench_phonons",
                        "source_row_id": material.row_id,
                    },
                )
            )
        output[split] = records
    return output


def _fit_property_ensemble(materials: list[ExternalMaterial], *, seed: int):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    matrix = np.asarray([material.vector for material in materials], dtype=float)
    targets = np.asarray([material.target for material in materials], dtype=float)
    models = []
    for replica in range(3):
        model = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=4,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.85,
            random_state=seed * 1009 + replica * 97,
            n_jobs=-1,
        )
        model.fit(matrix, targets)
        models.append(model)
    return tuple(models)


def _predict_ensemble(models, materials: list[ExternalMaterial]) -> list[tuple[float, float]]:
    import numpy as np

    if not materials:
        return []
    matrix = np.asarray([material.vector for material in materials], dtype=float)
    predictions = np.asarray([model.predict(matrix) for model in models], dtype=float)
    return [
        (float(predictions[:, index].mean()), float(predictions[:, index].var()))
        for index in range(len(materials))
    ]


@dataclass(frozen=True)
class _NoveltyReference:
    scaler: Any
    neighbors: Any
    training_distances: tuple[float, ...]
    row_ids: tuple[int, ...]

    @classmethod
    def fit(cls, materials: list[ExternalMaterial]) -> "_NoveltyReference":
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        matrix = np.asarray([material.vector for material in materials], dtype=float)
        scaler = StandardScaler().fit(matrix)
        standardized = scaler.transform(matrix)
        neighbors = NearestNeighbors(n_neighbors=2).fit(standardized)
        distances = neighbors.kneighbors(standardized)[0][:, 1]
        return cls(
            scaler=scaler,
            neighbors=neighbors,
            training_distances=tuple(sorted(float(value) for value in distances)),
            row_ids=tuple(material.row_id for material in materials),
        )

    def percentile(self, material: ExternalMaterial, *, exclude_self: bool) -> float:
        import numpy as np

        vector = self.scaler.transform(np.asarray([material.vector], dtype=float))
        neighbor_count = 2 if exclude_self and material.row_id in self.row_ids else 1
        distances = self.neighbors.kneighbors(vector, n_neighbors=neighbor_count)[0][0]
        distance = float(distances[-1])
        return _empirical_percentile(list(self.training_distances), distance)


def _coarse_novelty(
    material: ExternalMaterial, training_materials: list[ExternalMaterial]
) -> float:
    element_count = len(material.structure.composition.elements)
    training_counts = sorted(
        len(candidate.structure.composition.elements) for candidate in training_materials
    )
    return _empirical_percentile(training_counts, element_count)


def _structure_vector(structure) -> list[float]:
    fractions = [0.0] * 118
    composition = structure.composition.fractional_composition
    for element, amount in composition.items():
        fractions[int(element.Z) - 1] = float(amount)
    site_count = max(1, len(structure))
    lattice = structure.lattice
    return [
        *fractions,
        len(composition.elements) / 20.0,
        min(1.0, site_count / 100.0),
        min(5.0, float(structure.density)) / 5.0,
        min(100.0, float(structure.volume) / site_count) / 100.0,
        min(30.0, float(lattice.a)) / 30.0,
        min(30.0, float(lattice.b)) / 30.0,
        min(30.0, float(lattice.c)) / 30.0,
        float(lattice.alpha) / 180.0,
        float(lattice.beta) / 180.0,
        float(lattice.gamma) / 180.0,
    ]


def _structure_summary(structure) -> str:
    return (
        f"formula={structure.composition.reduced_formula}; "
        f"chemical_system={structure.composition.chemical_system}; "
        f"sites={len(structure)}; density={float(structure.density):.3f}"
    )


def _common_features(
    *, margin: float, uncertainty: float, cost: float, reversibility: float
) -> dict[str, float]:
    standardized_margin = margin / max(1e-6, uncertainty + 1e-3)
    if standardized_margin >= 0.0:
        confidence = 1.0 / (1.0 + math.exp(-min(60.0, standardized_margin)))
    else:
        exponential = math.exp(max(-60.0, standardized_margin))
        confidence = exponential / (1.0 + exponential)
    return {
        "verbal_confidence": confidence,
        "evidence_support": confidence,
        "evidence_conflict": min(1.0, uncertainty / (abs(margin) + uncertainty + 1e-6)),
        "perturbation_stability": 1.0 / (1.0 + uncertainty),
        "ood_score": min(1.0, uncertainty / (uncertainty + 1.0)),
        "source_reliability": 0.85,
        "surrogate_margin": margin,
        "surrogate_uncertainty": uncertainty,
        "cost": cost,
        "reversibility": reversibility,
    }


def _empirical_percentile(values: list[float], value: float) -> float:
    ordered = values if values == sorted(values) else sorted(values)
    return (bisect.bisect_right(ordered, value) + 0.5) / (len(ordered) + 1.0)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _validate_fold(
    records: dict[str, dict[str, list[ActionRecord]]],
    groups_by_split: dict[str, tuple[str, ...]],
) -> None:
    split_sets = {split: set(groups) for split, groups in groups_by_split.items()}
    for left_name, left_groups in split_sets.items():
        for right_name, right_groups in split_sets.items():
            if left_name >= right_name:
                continue
            if left_groups & right_groups:
                raise ValueError(f"group leakage between {left_name} and {right_name}")
    for task, split_records in records.items():
        if any(not task_records for task_records in split_records.values()):
            raise ValueError(f"task {task} has an empty split")
        record_ids = [
            record.record_id for task_records in split_records.values() for record in task_records
        ]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"task {task} contains duplicate record IDs")
