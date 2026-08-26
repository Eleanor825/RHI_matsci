from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .external_h14 import balanced_group_folds, weighted_group_partition
from .schema import ActionRecord


TASK = "external_extreme_superconductivity"


@dataclass(frozen=True)
class SuperconductivityMaterial:
    row_id: int
    composition: str
    target: float
    group_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SuperconductivityFold:
    fold: int
    records_by_split: dict[str, list[ActionRecord]]
    groups_by_split: dict[str, tuple[str, ...]]


def load_external_h23_materials(source_path: str | Path) -> list[SuperconductivityMaterial]:
    from pymatgen.core import Composition

    with gzip.open(source_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    columns = {name: index for index, name in enumerate(payload["columns"])}
    output = []
    for row_id, row in zip(payload["index"], payload["data"]):
        try:
            composition = Composition(row[columns["composition"]])
        except (TypeError, ValueError):
            continue
        target = float(row[columns["Tc"]])
        if not math.isfinite(target):
            continue
        output.append(
            SuperconductivityMaterial(
                row_id=int(row_id),
                composition=str(composition.reduced_formula),
                target=target,
                group_id=str(composition.chemical_system),
                vector=composition_feature_vector(composition),
            )
        )
    return output


def composition_feature_vector(composition) -> tuple[float, ...]:
    fractions = [0.0] * 118
    elements = []
    weights = []
    for element, amount in composition.fractional_composition.items():
        fraction = float(amount)
        fractions[element.Z - 1] = fraction
        elements.append(element)
        weights.append(fraction)
    output = list(fractions)
    scales = (100.0, 250.0, 4.0, 7.0, 18.0)
    properties = (
        [float(element.Z) for element in elements],
        [float(element.atomic_mass) for element in elements],
        [float(element.X or 0.0) for element in elements],
        [float(element.row or 0.0) for element in elements],
        [float(element.group or 0.0) for element in elements],
    )
    for values, scale in zip(properties, scales):
        mean = sum(weight * value for weight, value in zip(weights, values))
        variance = sum(
            weight * (value - mean) ** 2 for weight, value in zip(weights, values)
        )
        output.extend((mean / scale, math.sqrt(variance) / scale))
    entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
    output.extend((len(elements) / 10.0, entropy / math.log(20.0)))
    return tuple(output)


def build_external_h23_fold(
    source_path: str | Path,
    *,
    fold: int,
    n_folds: int = 5,
    seed: int = 2301,
) -> SuperconductivityFold:
    materials = load_external_h23_materials(source_path)
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
        split: [
            material for material in materials if material.group_id in set(group_ids)
        ]
        for split, group_ids in groups.items()
    }
    records = _build_records(splits, seed=seed + fold)
    _validate(records, groups)
    return SuperconductivityFold(fold, records, groups)


def write_external_h23_fold(
    fold: SuperconductivityFold,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_root) / f"fold_{fold.fold}"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, records in fold.records_by_split.items():
        path = root / f"{TASK}.{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(record.to_json(), sort_keys=True) for record in records)
            + "\n"
        )
        counts[split] = len(records)
    manifest = {
        "fold": fold.fold,
        "task": TASK,
        "counts": counts,
        "groups_by_split": {
            split: list(values) for split, values in fold.groups_by_split.items()
        },
        "group_disjoint": True,
        "utility": "train-empirical percentile of experimental critical temperature",
        "extreme_definition": "top decile of train experimental critical temperature",
        "hidden_outcome_contract": True,
        "source_cleaning": {
            "rule": "drop rows whose composition is not parseable by pymatgen or whose Tc is non-finite",
            "retained_rows": sum(counts.values()),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _build_records(splits, *, seed):
    train_targets = sorted(material.target for material in splits["train"])
    threshold = _quantile(train_targets, 0.90)
    models = _fit_ensemble(splits["train"], seed=seed)
    novelty = _NoveltyReference.fit(splits["train"])
    output = {}
    for split, materials in splits.items():
        predictions = _predict_ensemble(models, materials)
        rows = []
        for material, (prediction, variance) in zip(materials, predictions):
            standard_deviation = math.sqrt(max(1e-9, variance))
            margin = prediction - threshold
            confidence = _logistic(margin / (standard_deviation + 1e-3))
            utility = _percentile(train_targets, material.target)
            ood = novelty.percentile(material, exclude_self=split == "train")
            rows.append(
                ActionRecord(
                    record_id=f"external-superconductivity-extreme::{material.row_id}",
                    benchmark=TASK,
                    split=split,
                    visible_context=(
                        "Screen a composition for an extreme high superconducting critical temperature. "
                        f"Candidate composition: {material.composition}. The measured critical temperature "
                        "is withheld until post-action evaluation."
                    ),
                    candidate_action=(
                        "Advance this composition for high-Tc superconductivity validation."
                    ),
                    action_type="choose_candidate",
                    evidence=[
                        f"train-only predicted Tc={prediction:.6f}",
                        f"ensemble standard deviation={standard_deviation:.6f}",
                    ],
                    features={
                        "verbal_confidence": confidence,
                        "evidence_support": _percentile(train_targets, prediction),
                        "evidence_conflict": 1.0 - confidence,
                        "perturbation_stability": 1.0 / (1.0 + standard_deviation),
                        "ood_score": ood,
                        "source_reliability": 0.88,
                        "cost": 0.50,
                        "reversibility": 0.60,
                    },
                    label=int(material.target >= threshold),
                    utility=utility,
                    metadata={
                        "group_id": material.group_id,
                        "source_dataset": "superconductivity2018",
                        "source_row_id": material.row_id,
                        "composition": material.composition,
                        "train_tc_extreme_threshold": threshold,
                    },
                )
            )
        output[split] = rows
    return output


@dataclass(frozen=True)
class _NoveltyReference:
    scaler: Any
    neighbors: Any
    distances: tuple[float, ...]

    @classmethod
    def fit(cls, materials):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        matrix = np.asarray([material.vector for material in materials], dtype=float)
        scaler = StandardScaler().fit(matrix)
        transformed = scaler.transform(matrix)
        neighbors = NearestNeighbors(n_neighbors=2).fit(transformed)
        distances = neighbors.kneighbors(transformed)[0][:, 1]
        return cls(scaler, neighbors, tuple(sorted(float(value) for value in distances)))

    def percentile(self, material, *, exclude_self):
        transformed = self.scaler.transform([material.vector])
        count = 2 if exclude_self else 1
        distance = float(self.neighbors.kneighbors(transformed, n_neighbors=count)[0][0][-1])
        return _percentile(self.distances, distance)


def _fit_ensemble(materials, *, seed):
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
    return list(zip(predictions.mean(axis=0), predictions.var(axis=0)))


def _percentile(values, value):
    import bisect

    ordered = values if list(values) == sorted(values) else sorted(values)
    return (bisect.bisect_right(ordered, value) + 0.5) / (len(ordered) + 1.0)


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
        return 1.0 / (1.0 + math.exp(-min(60.0, value)))
    exponential = math.exp(max(-60.0, value))
    return exponential / (1.0 + exponential)


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
