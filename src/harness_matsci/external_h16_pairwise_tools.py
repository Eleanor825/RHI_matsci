from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .external_h15_unique import (
    UniqueMaterial,
    load_external_h15_unique_materials,
)
from .external_h16_pairwise import TASK
from .schema import ActionRecord


EVIDENCE_SOURCES = ("composition", "structure", "high_fidelity")


@dataclass(frozen=True)
class _Regressor:
    model: Any
    residual_scale: float
    feature_slice: tuple[int, int]


class ExternalH16PairwiseEvidenceEnvironment:
    def __init__(self, material_by_id, regressors, train_targets):
        self.material_by_id = material_by_id
        self.regressors = regressors
        self.train_targets = train_targets

    @classmethod
    def fit(
        cls,
        source_path: str | Path,
        train_records: list[ActionRecord],
        *,
        seed: int,
    ) -> "ExternalH16PairwiseEvidenceEnvironment":
        materials = load_external_h15_unique_materials(source_path)
        by_id = {material.row_id: material for material in materials}
        train_ids = {
            int(record.metadata[key])
            for record in train_records
            for key in ("chosen_row_id", "alternative_row_id")
        }
        train_materials = [by_id[row_id] for row_id in sorted(train_ids)]
        regressors = {
            source: _fit_regressor(train_materials, source=source, seed=seed + index * 97)
            for index, source in enumerate(("composition", "structure"))
        }
        return cls(
            by_id,
            regressors,
            tuple(sorted(material.target for material in train_materials)),
        )

    def observe(self, record: ActionRecord, *, source: str) -> ActionRecord:
        return self.observe_many([record], source=source)[0]

    def observe_many(self, records: list[ActionRecord], *, source: str):
        if not records:
            return []
        if source not in EVIDENCE_SOURCES:
            raise ValueError(f"unknown source {source}")
        chosen = [
            self.material_by_id[int(record.metadata["chosen_row_id"])]
            for record in records
        ]
        alternatives = [
            self.material_by_id[int(record.metadata["alternative_row_id"])]
            for record in records
        ]
        if source == "high_fidelity":
            chosen_values = [material.target for material in chosen]
            alternative_values = [material.target for material in alternatives]
            residual = max(1e-6, 0.01 * _robust_scale(self.train_targets))
            residuals = [residual] * len(records)
            tool = "costed_database_measurement_replay"
        else:
            regressor = self.regressors[source]
            chosen_values = _predict_many(regressor, chosen)
            alternative_values = _predict_many(regressor, alternatives)
            residuals = [regressor.residual_scale] * len(records)
            tool = "train_only_bulk_modulus_pair_model"
        return [
            self._build_view(
                record,
                source=source,
                chosen_value=chosen_value,
                alternative_value=alternative_value,
                residual=residual,
                tool=tool,
            )
            for record, chosen_value, alternative_value, residual in zip(
                records, chosen_values, alternative_values, residuals
            )
        ]

    def _build_view(
        self,
        record,
        *,
        source,
        chosen_value,
        alternative_value,
        residual,
        tool,
    ):
        utility = _pair_utility(self.train_targets, chosen_value, alternative_value)
        utility_low = _pair_utility(
            self.train_targets,
            chosen_value - residual,
            alternative_value + residual,
        )
        utility_high = _pair_utility(
            self.train_targets,
            chosen_value + residual,
            alternative_value - residual,
        )
        margin = chosen_value - alternative_value
        uncertainty = math.sqrt(2.0) * residual
        scale = uncertainty + 1e-3
        confidence = _logistic(margin / scale)
        confidence_low = _logistic((margin - uncertainty) / scale)
        confidence_high = _logistic((margin + uncertainty) / scale)
        tool_cost = 0.020 if source == "high_fidelity" else 0.002
        return replace(
            record,
            evidence=[
                *record.evidence,
                f"{source} pair margin={margin:.6f}",
                f"{source} estimated pair utility={utility:.6f}",
            ],
            features={
                **record.features,
                "tool_estimated_margin": margin,
                "tool_uncertainty": uncertainty,
                "tool_source_confidence": confidence,
                "tool_source_confidence_low": confidence_low,
                "tool_source_confidence_high": confidence_high,
                "tool_estimated_utility": utility,
                "tool_estimated_utility_low": min(utility, utility_low),
                "tool_estimated_utility_high": max(utility, utility_high),
                "tool_source_composition": float(source == "composition"),
                "tool_source_structure": float(source == "structure"),
                "tool_source_high_fidelity": float(source == "high_fidelity"),
                "tool_cost": tool_cost,
            },
            metadata={
                **record.metadata,
                "evidence_source": source,
                "evidence_provenance": {
                    "source_family": source,
                    "tool": tool,
                    "target_label_exposed": False,
                },
                "hidden_outcome_exposed": False,
                "outcome_defining_measurement_replayed": source == "high_fidelity",
            },
        )


def _feature_slice(source):
    if source == "composition":
        return 0, 118
    if source == "structure":
        return 118, 126
    raise ValueError(source)


def _fit_regressor(materials: list[UniqueMaterial], *, source, seed):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    start, end = _feature_slice(source)
    matrix = np.asarray([material.vector[start:end] for material in materials], dtype=float)
    targets = np.asarray([material.target for material in materials], dtype=float)
    model = ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=5,
        max_features=0.8,
        bootstrap=True,
        max_samples=0.85,
        random_state=seed,
        n_jobs=-1,
    ).fit(matrix, targets)
    residual = float(np.sqrt(np.mean((targets - model.predict(matrix)) ** 2)))
    return _Regressor(model, max(1e-6, residual), (start, end))


def _predict_many(regressor, materials):
    start, end = regressor.feature_slice
    matrix = [material.vector[start:end] for material in materials]
    return [float(value) for value in regressor.model.predict(matrix)]


def _pair_utility(train_targets, chosen, alternative):
    return max(
        0.0,
        _empirical_percentile(train_targets, chosen)
        - _empirical_percentile(train_targets, alternative),
    )


def _empirical_percentile(values, value):
    import bisect

    ordered = values if list(values) == sorted(values) else sorted(values)
    return (bisect.bisect_right(ordered, value) + 0.5) / (len(ordered) + 1.0)


def _robust_scale(values):
    ordered = sorted(values)
    return max(1e-6, _quantile(ordered, 0.75) - _quantile(ordered, 0.25))


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
