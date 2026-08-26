from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .external_h15_unique import (
    TASK,
    UniqueMaterial,
    load_external_h15_unique_materials,
)
from .schema import ActionRecord


EVIDENCE_SOURCES = ("composition", "structure", "high_fidelity")


@dataclass(frozen=True)
class _Regressor:
    model: Any
    residual_scale: float
    feature_slice: tuple[int, int]


@dataclass(frozen=True)
class _Novelty:
    scaler: Any
    neighbors: Any
    distances: tuple[float, ...]
    feature_slice: tuple[int, int]


class ExternalH15UniqueEvidenceEnvironment:
    def __init__(
        self,
        material_by_id: dict[int, UniqueMaterial],
        regressors: dict[str, _Regressor],
        novelty: dict[str, _Novelty],
        train_targets: tuple[float, ...],
        source_thresholds: dict[str, float],
    ) -> None:
        self.material_by_id = material_by_id
        self.regressors = regressors
        self.novelty = novelty
        self.train_targets = train_targets
        self.source_thresholds = source_thresholds

    @classmethod
    def fit(
        cls,
        source_path: str | Path,
        train_records: list[ActionRecord],
        *,
        seed: int,
    ) -> "ExternalH15UniqueEvidenceEnvironment":
        materials = load_external_h15_unique_materials(source_path)
        by_id = {material.row_id: material for material in materials}
        train_materials = [
            by_id[int(record.metadata["source_row_id"])] for record in train_records
        ]
        regressors = {
            source: _fit_regressor(train_materials, source=source, seed=seed + index * 97)
            for index, source in enumerate(("composition", "structure"))
        }
        novelty = {
            source: _fit_novelty(train_materials, source=source)
            for source in ("composition", "structure")
        }
        novelty["high_fidelity"] = _fit_novelty(train_materials, source="combined")
        train_targets = tuple(sorted(material.target for material in train_materials))
        source_thresholds = {}
        for source in EVIDENCE_SOURCES:
            predictions = (
                [material.target for material in train_materials]
                if source == "high_fidelity"
                else _predict_many(regressors[source], train_materials)
            )
            uniqueness_values = _novelty_percentiles(
                novelty[source], train_materials, exclude_self=True
            )
            utilities = [
                _utility(train_targets, prediction, uniqueness)
                for prediction, uniqueness in zip(predictions, uniqueness_values)
            ]
            source_thresholds[source] = _quantile(utilities, 0.90)
        return cls(by_id, regressors, novelty, train_targets, source_thresholds)

    def observe(self, record: ActionRecord, *, source: str) -> ActionRecord:
        return self.observe_many([record], source=source)[0]

    def observe_many(
        self, records: list[ActionRecord], *, source: str
    ) -> list[ActionRecord]:
        if not records:
            return []
        if any(record.benchmark != TASK for record in records):
            raise ValueError("all records must belong to the H15 unique task")
        if source not in EVIDENCE_SOURCES:
            raise ValueError(f"unknown source {source}")
        materials = [
            self.material_by_id[int(record.metadata["source_row_id"])]
            for record in records
        ]
        if source == "high_fidelity":
            predictions = [material.target for material in materials]
            residual = max(1e-6, 0.01 * _robust_scale(self.train_targets))
            residuals = [residual] * len(materials)
            tool = "costed_database_measurement_replay"
        else:
            regressor = self.regressors[source]
            predictions = _predict_many(regressor, materials)
            residuals = [regressor.residual_scale] * len(materials)
            tool = "train_only_bulk_modulus_model"
        uniqueness_values = _novelty_percentiles(
            self.novelty[source], materials, exclude_self=False
        )
        return [
            self._build_view(
                record,
                source=source,
                prediction=prediction,
                residual=residual,
                uniqueness=uniqueness,
                tool=tool,
            )
            for record, prediction, residual, uniqueness in zip(
                records, predictions, residuals, uniqueness_values
            )
        ]

    def _build_view(
        self,
        record: ActionRecord,
        *,
        source: str,
        prediction: float,
        residual: float,
        uniqueness: float,
        tool: str,
    ) -> ActionRecord:
        if record.benchmark != TASK:
            raise ValueError(f"unsupported task {record.benchmark}")
        utility = _utility(self.train_targets, prediction, uniqueness)
        utility_low = _utility(self.train_targets, prediction - residual, uniqueness)
        utility_high = _utility(self.train_targets, prediction + residual, uniqueness)
        margin = utility - self.source_thresholds[source]
        uncertainty = max(1e-6, 0.5 * (utility_high - utility_low))
        scale = uncertainty + 1e-3
        confidence = _logistic(margin / scale)
        confidence_low = _logistic((margin - uncertainty) / scale)
        confidence_high = _logistic((margin + uncertainty) / scale)
        tool_cost = 0.020 if source == "high_fidelity" else 0.002
        return replace(
            record,
            evidence=[
                *record.evidence,
                f"{source} estimated discovery utility={utility:.6f}",
                f"{source} utility uncertainty={uncertainty:.6f}",
            ],
            features={
                **record.features,
                "tool_estimated_margin": margin,
                "tool_uncertainty": uncertainty,
                "tool_source_confidence": confidence,
                "tool_source_confidence_low": confidence_low,
                "tool_source_confidence_high": confidence_high,
                "tool_estimated_utility": utility,
                "tool_estimated_utility_low": utility_low,
                "tool_estimated_utility_high": utility_high,
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


def _feature_slice(source: str) -> tuple[int, int]:
    if source == "composition":
        return 0, 118
    if source == "structure":
        return 118, 126
    if source == "combined":
        return 0, 126
    raise ValueError(source)


def _fit_regressor(materials, *, source, seed):
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


def _predict(regressor, material):
    start, end = regressor.feature_slice
    return float(regressor.model.predict([material.vector[start:end]])[0])


def _predict_many(regressor, materials):
    start, end = regressor.feature_slice
    matrix = [material.vector[start:end] for material in materials]
    return [float(value) for value in regressor.model.predict(matrix)]


def _fit_novelty(materials, *, source):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    start, end = _feature_slice(source)
    matrix = np.asarray([material.vector[start:end] for material in materials], dtype=float)
    scaler = StandardScaler().fit(matrix)
    transformed = scaler.transform(matrix)
    neighbors = NearestNeighbors(n_neighbors=2).fit(transformed)
    distances = neighbors.kneighbors(transformed)[0][:, 1]
    return _Novelty(
        scaler,
        neighbors,
        tuple(sorted(float(value) for value in distances)),
        (start, end),
    )


def _novelty_percentile(novelty, material, *, exclude_self):
    start, end = novelty.feature_slice
    transformed = novelty.scaler.transform([material.vector[start:end]])
    count = 2 if exclude_self else 1
    distance = float(novelty.neighbors.kneighbors(transformed, n_neighbors=count)[0][0][-1])
    return _empirical_percentile(list(novelty.distances), distance)


def _novelty_percentiles(novelty, materials, *, exclude_self):
    start, end = novelty.feature_slice
    matrix = [material.vector[start:end] for material in materials]
    transformed = novelty.scaler.transform(matrix)
    count = 2 if exclude_self else 1
    distances = novelty.neighbors.kneighbors(
        transformed, n_neighbors=count
    )[0][:, -1]
    return [
        _empirical_percentile(list(novelty.distances), float(distance))
        for distance in distances
    ]


def _utility(train_targets, prediction, uniqueness):
    return 0.5 * (_empirical_percentile(list(train_targets), prediction) + uniqueness)


def _empirical_percentile(values, value):
    import bisect

    ordered = values if values == sorted(values) else sorted(values)
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


def _robust_scale(values):
    ordered = sorted(values)
    return max(1e-6, _quantile(ordered, 0.75) - _quantile(ordered, 0.25))


def _logistic(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(60.0, value)))
    exponential = math.exp(max(-60.0, value))
    return exponential / (1.0 + exponential)
