from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .external_h14 import balanced_group_folds, weighted_group_partition
from .matbench_pairwise import load_matbench_log_kvrh_rows
from .schema import ActionRecord


TASK = "external_unique_log_kvrh"
CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


@dataclass(frozen=True)
class UniqueMaterial:
    row_id: int
    mbid: str
    composition: str
    crystal_system: str
    space_group: int
    target: float
    group_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class UniqueFold:
    fold: int
    records_by_split: dict[str, list[ActionRecord]]
    groups_by_split: dict[str, tuple[str, ...]]


def build_external_h15_unique_fold(
    source_path: str | Path,
    *,
    fold: int,
    n_folds: int = 5,
    seed: int = 1501,
) -> UniqueFold:
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
    records = _build_records(splits, seed=seed + fold)
    _validate(records, groups)
    return UniqueFold(fold, records, groups)


def write_external_h15_unique_fold(
    fold: UniqueFold, output_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root) / f"fold_{fold.fold}"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, records in fold.records_by_split.items():
        path = root / f"{TASK}.{split}.jsonl"
        path.write_text(
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
        "group_disjoint": True,
        "utility": "0.5 * train-percentile bulk-modulus performance + 0.5 * train-reference composition/structure novelty",
        "hidden_outcome_contract": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_external_h15_unique_materials(
    source_path: str | Path,
) -> list[UniqueMaterial]:
    from pymatgen.core import Composition

    rows = load_matbench_log_kvrh_rows(source_path=source_path)
    output = []
    for row_id, row in enumerate(rows):
        composition = Composition(row["composition"])
        fractions = [0.0] * 118
        for element, amount in composition.fractional_composition.items():
            fractions[element.Z - 1] = float(amount)
        system = str(row["crys_sys"]).lower()
        system_vector = [float(system == value) for value in CRYSTAL_SYSTEMS]
        vector = tuple(
            fractions
            + [float(row["spg_num"]) / 230.0]
            + system_vector
        )
        output.append(
            UniqueMaterial(
                row_id=row_id,
                mbid=str(row["mbid"]),
                composition=str(row["composition"]),
                crystal_system=system,
                space_group=int(row["spg_num"]),
                target=float(row["log10_k_vrh"]),
                group_id=str(composition.chemical_system),
                vector=vector,
            )
        )
    return output


def _build_records(
    splits: dict[str, list[UniqueMaterial]], *, seed: int
) -> dict[str, list[ActionRecord]]:
    train_targets = sorted(material.target for material in splits["train"])
    novelty = _NoveltyReference.fit(splits["train"])
    model = _fit_regressor(splits["train"], seed=seed)
    train_scores = [
        0.5 * _empirical_percentile(train_targets, material.target)
        + 0.5 * novelty.percentile(material, exclude_self=True)
        for material in splits["train"]
    ]
    threshold = _quantile(train_scores, 0.90)
    output = {}
    for split, materials in splits.items():
        predictions = _predict_ensemble(model, materials)
        rows = []
        for material, (prediction, variance) in zip(materials, predictions):
            performance = _empirical_percentile(train_targets, material.target)
            uniqueness = novelty.percentile(
                material, exclude_self=split == "train"
            )
            utility = 0.5 * (performance + uniqueness)
            surrogate_performance = _empirical_percentile(train_targets, prediction)
            surrogate_utility = 0.5 * (surrogate_performance + uniqueness)
            rows.append(
                ActionRecord(
                    record_id=f"external-log-kvrh-unique::{material.mbid}",
                    benchmark=TASK,
                    split=split,
                    visible_context=(
                        "DiSCoVeR-style Materials Project screening for high bulk modulus "
                        "and chemical/structural uniqueness. "
                        f"Composition: {material.composition}; crystal system: "
                        f"{material.crystal_system}; space group: {material.space_group}."
                    ),
                    candidate_action="Advance this material for unique-discovery validation.",
                    action_type="choose_candidate",
                    evidence=[
                        f"train-only discovery utility estimate={surrogate_utility:.6f}",
                        f"property ensemble disagreement={math.sqrt(max(0.0, variance)):.6f}",
                    ],
                    features={
                        "verbal_confidence": surrogate_utility,
                        "evidence_support": surrogate_performance,
                        "evidence_conflict": 1.0 - surrogate_performance,
                        "perturbation_stability": max(0.0, 1.0 - math.sqrt(max(0.0, variance))),
                        "ood_score": uniqueness,
                        "source_reliability": 0.88,
                        "cost": 0.50,
                        "reversibility": 0.60,
                    },
                    label=int(utility >= threshold),
                    utility=utility,
                    metadata={
                        "group_id": material.group_id,
                        "source_dataset": "matbench_log_kvrh",
                        "source_row_id": material.row_id,
                        "mbid": material.mbid,
                        "composition": material.composition,
                        "crystal_system": material.crystal_system,
                        "space_group": material.space_group,
                    },
                )
            )
        output[split] = rows
    return output


@dataclass(frozen=True)
class _NoveltyReference:
    scaler: Any
    neighbors: Any
    training_distances: tuple[float, ...]

    @classmethod
    def fit(cls, materials: list[UniqueMaterial]) -> "_NoveltyReference":
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        matrix = np.asarray([material.vector for material in materials], dtype=float)
        scaler = StandardScaler().fit(matrix)
        transformed = scaler.transform(matrix)
        neighbors = NearestNeighbors(n_neighbors=2).fit(transformed)
        distances = neighbors.kneighbors(transformed)[0][:, 1]
        return cls(scaler, neighbors, tuple(sorted(float(value) for value in distances)))

    def percentile(self, material: UniqueMaterial, *, exclude_self: bool) -> float:
        transformed = self.scaler.transform([material.vector])
        count = 2 if exclude_self else 1
        distance = float(
            self.neighbors.kneighbors(transformed, n_neighbors=count)[0][0][-1]
        )
        return _empirical_percentile(list(self.training_distances), distance)


def _fit_regressor(materials: list[UniqueMaterial], *, seed: int):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    matrix = np.asarray([material.vector for material in materials], dtype=float)
    targets = np.asarray([material.target for material in materials], dtype=float)
    models = []
    for replica in range(3):
        models.append(
            ExtraTreesRegressor(
                n_estimators=160,
                min_samples_leaf=5,
                max_features=0.8,
                bootstrap=True,
                max_samples=0.85,
                random_state=seed * 1009 + replica * 97,
                n_jobs=-1,
            ).fit(matrix, targets)
        )
    return tuple(models)


def _predict_ensemble(models, materials):
    import numpy as np

    matrix = np.asarray([material.vector for material in materials], dtype=float)
    predictions = np.asarray([model.predict(matrix) for model in models], dtype=float)
    return list(zip(np.mean(predictions, axis=0), np.var(predictions, axis=0)))


def _empirical_percentile(values: list[float], value: float) -> float:
    import bisect

    ordered = values if values == sorted(values) else sorted(values)
    return (bisect.bisect_right(ordered, value) + 0.5) / (len(ordered) + 1.0)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate(records, groups):
    sets = {split: set(values) for split, values in groups.items()}
    names = tuple(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError(f"group leakage between {left} and {right}")
    for split, rows in records.items():
        if not rows:
            raise ValueError(f"empty split {split}")
        if any(row.metadata["group_id"] not in sets[split] for row in rows):
            raise ValueError(f"record group mismatch in {split}")
