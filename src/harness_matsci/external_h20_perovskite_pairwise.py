from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matminer.utils.io import load_dataframe_from_json

from .external_h14 import (
    ExternalMaterial,
    _fit_property_ensemble,
    _predict_ensemble,
    _structure_summary,
    _structure_vector,
    balanced_group_folds,
    weighted_group_partition,
)
from .external_h14_tools import _fit_source_regressor, _predict_source
from .external_h17_pairwise import REGIME_PROPORTIONS, _make_mixed_pairs
from .schema import ActionRecord


TASK = "external_pairwise_mixed_perovskite_stability"
DATASET = "matbench_perovskites"
FILENAME = "matbench_perovskites.json.gz"


@dataclass(frozen=True)
class _MaterialAdapter:
    material: ExternalMaterial

    @property
    def row_id(self):
        return self.material.row_id

    @property
    def mbid(self):
        return str(self.material.row_id)

    @property
    def target(self):
        return self.material.target

    @property
    def group_id(self):
        return self.material.group_id

    @property
    def vector(self):
        return self.material.vector

    @property
    def composition(self):
        return self.material.structure.composition.reduced_formula

    @property
    def crystal_system(self):
        return "structure-derived"

    @property
    def space_group(self):
        return "withheld"


@dataclass(frozen=True)
class PerovskitePairwiseFold:
    fold: int
    records_by_split: dict[str, list[ActionRecord]]
    groups_by_split: dict[str, tuple[str, ...]]


def load_external_h20_materials(snapshot_root: str | Path) -> list[ExternalMaterial]:
    frame = load_dataframe_from_json(Path(snapshot_root) / FILENAME, pbar=False, decode=True)
    materials = []
    for row_id, row in frame.iterrows():
        structure = row["structure"]
        materials.append(
            ExternalMaterial(
                dataset=DATASET,
                row_id=int(row_id),
                structure=structure,
                target=-float(row["e_form"]),
                group_id=str(structure.composition.chemical_system),
                vector=tuple(_structure_vector(structure)),
            )
        )
    return materials


def build_external_h20_perovskite_fold(
    snapshot_root: str | Path,
    *,
    fold: int,
    n_folds: int = 5,
    seed: int = 2001,
    pair_multiplier: float = 1.0,
) -> PerovskitePairwiseFold:
    materials = load_external_h20_materials(snapshot_root)
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
    return PerovskitePairwiseFold(fold, records, groups)


def write_external_h20_perovskite_fold(
    fold: PerovskitePairwiseFold, output_root: str | Path
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
        "source_dataset": DATASET,
        "original_target": "e_form",
        "scientific_score": "negative formation energy",
        "pair_multiplier": 1.0,
        "pair_generation": "train-only easy/ambiguous/OOD surrogate stratification",
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
    ensemble = _fit_property_ensemble(train, seed=seed)
    composition = _fit_source_regressor(train, source="composition", seed=seed + 97)
    structure = _fit_source_regressor(train, source="structure", seed=seed + 193)
    output = {}
    for split_index, (split, materials) in enumerate(splits.items()):
        adapters = [_MaterialAdapter(material) for material in materials]
        desired = max(1, int(round(len(materials) * pair_multiplier)))
        rows = _make_mixed_pairs(
            adapters,
            combined_predictions=_predict_ensemble(ensemble, materials),
            composition_predictions=[_predict_source(composition, row) for row in materials],
            structure_predictions=[_predict_source(structure, row) for row in materials],
            train_targets=train_targets,
            desired=desired,
            seed=seed + split_index * 503,
            split=split,
        )
        by_id = {material.row_id: material for material in materials}
        output[split] = [_convert_record(row, by_id=by_id, split=split) for row in rows]
    return output


def _convert_record(record, *, by_id, split):
    chosen_id = int(record.metadata["chosen_row_id"])
    alternative_id = int(record.metadata["alternative_row_id"])
    chosen = by_id[chosen_id]
    alternative = by_id[alternative_id]
    regime = record.metadata["pair_regime"]
    return ActionRecord(
        record_id=f"external-perovskite-mixed::{split}::{regime}::{chosen_id}::{alternative_id}",
        benchmark=TASK,
        split=split,
        visible_context=(
            "Choose the perovskite expected to have the lower formation energy. "
            f"Candidate A: {_structure_summary(chosen.structure)}. Candidate B: "
            f"{_structure_summary(alternative.structure)}."
        ),
        candidate_action="Choose candidate A for perovskite stability follow-up.",
        action_type=record.action_type,
        evidence=record.evidence,
        features=record.features,
        label=record.label,
        utility=record.utility,
        metadata={
            **record.metadata,
            "source_dataset": DATASET,
            "original_target": "e_form",
            "scientific_score": "negative formation energy",
            "chosen_row_id": chosen_id,
            "alternative_row_id": alternative_id,
        },
    )


def _validate(records, groups):
    sets = {split: set(values) for split, values in groups.items()}
    names = tuple(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError(f"group leakage between {left} and {right}")
    for split, rows in records.items():
        if set(row.metadata["pair_regime"] for row in rows) != set(REGIME_PROPORTIONS):
            raise ValueError(f"missing regime in {split}")
        for row in rows:
            if row.metadata["chosen_group_id"] not in sets[split]:
                raise ValueError(f"chosen group leakage in {split}")
            if row.metadata["alternative_group_id"] not in sets[split]:
                raise ValueError(f"alternative group leakage in {split}")
