from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .external_h23_superconductivity import (
    TASK,
    SuperconductivityMaterial,
    load_external_h23_materials,
)
from .schema import ActionRecord


EVIDENCE_SOURCES = ("coarse_composition", "full_composition", "high_fidelity")


@dataclass(frozen=True)
class _Regressor:
    model: Any
    residual_scale: float
    feature_slice: tuple[int, int]


class ExternalH23SuperconductivityEnvironment:
    def __init__(self, materials, regressors, train_targets, thresholds):
        self.material_by_id = {material.row_id: material for material in materials}
        self.regressors = regressors
        self.train_targets = train_targets
        self.thresholds = thresholds

    @classmethod
    def fit(cls, source_path: str | Path, train_records, *, seed: int):
        materials = load_external_h23_materials(source_path)
        by_id = {material.row_id: material for material in materials}
        train_materials = [
            by_id[int(record.metadata["source_row_id"])] for record in train_records
        ]
        regressors = {
            source: _fit_regressor(train_materials, source=source, seed=seed + index * 97)
            for index, source in enumerate(EVIDENCE_SOURCES[:-1])
        }
        train_targets = tuple(sorted(material.target for material in train_materials))
        thresholds = {}
        for source in EVIDENCE_SOURCES:
            predictions = (
                [material.target for material in train_materials]
                if source == "high_fidelity"
                else _predict_many(regressors[source], train_materials)
            )
            utilities = [_percentile(train_targets, value) for value in predictions]
            thresholds[source] = _quantile(utilities, 0.90)
        return cls(materials, regressors, train_targets, thresholds)

    def observe(self, record: ActionRecord, *, source: str) -> ActionRecord:
        return self.observe_many([record], source=source)[0]

    def observe_many(self, records, *, source):
        if not records:
            return []
        if source not in EVIDENCE_SOURCES:
            raise ValueError(f"unknown source {source}")
        if any(record.benchmark != TASK for record in records):
            raise ValueError("all records must belong to H23")
        materials = [
            self.material_by_id[int(record.metadata["source_row_id"])] for record in records
        ]
        if source == "high_fidelity":
            predictions = [material.target for material in materials]
            residuals = [max(1e-6, 0.01 * _robust_scale(self.train_targets))] * len(records)
            tool = "costed_experimental_tc_measurement_replay"
        else:
            regressor = self.regressors[source]
            predictions = _predict_many(regressor, materials)
            residuals = [regressor.residual_scale] * len(records)
            tool = "train_only_composition_tc_model"
        return [
            self._view(record, source, prediction, residual, tool)
            for record, prediction, residual in zip(records, predictions, residuals)
        ]

    def _view(self, record, source, prediction, residual, tool):
        utility = _percentile(self.train_targets, prediction)
        utility_low = _percentile(self.train_targets, prediction - residual)
        utility_high = _percentile(self.train_targets, prediction + residual)
        margin = utility - self.thresholds[source]
        uncertainty = max(1e-6, 0.5 * (utility_high - utility_low))
        confidence = _logistic(margin / (uncertainty + 1e-3))
        return replace(
            record,
            evidence=[
                *record.evidence,
                f"{source} estimated Tc percentile={utility:.6f}",
                f"{source} utility uncertainty={uncertainty:.6f}",
            ],
            features={
                **record.features,
                "tool_estimated_margin": margin,
                "tool_uncertainty": uncertainty,
                "tool_source_confidence": confidence,
                "tool_source_confidence_low": _logistic(
                    (margin - uncertainty) / (uncertainty + 1e-3)
                ),
                "tool_source_confidence_high": _logistic(
                    (margin + uncertainty) / (uncertainty + 1e-3)
                ),
                "tool_estimated_utility": utility,
                "tool_estimated_utility_low": utility_low,
                "tool_estimated_utility_high": utility_high,
                "tool_source_coarse_composition": float(source == "coarse_composition"),
                "tool_source_full_composition": float(source == "full_composition"),
                "tool_source_high_fidelity": float(source == "high_fidelity"),
                "tool_cost": 0.020 if source == "high_fidelity" else 0.002,
            },
            metadata={
                **record.metadata,
                "evidence_source": source,
                "evidence_provenance": {
                    "source_family": source,
                    "tool": tool,
                    "target_label_exposed": False,
                    "training_only": source != "high_fidelity",
                },
                "hidden_outcome_exposed": False,
                "outcome_defining_measurement_replayed": source == "high_fidelity",
            },
        )


def _feature_slice(source):
    if source == "coarse_composition":
        return 118, 130
    if source == "full_composition":
        return 0, 130
    raise ValueError(source)


def _fit_regressor(materials: list[SuperconductivityMaterial], *, source, seed):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    start, end = _feature_slice(source)
    ordered = sorted(
        materials,
        key=lambda material: hashlib.sha256(
            f"{seed}|{source}|{material.row_id}".encode()
        ).hexdigest(),
    )
    calibration_count = max(20, int(math.ceil(0.20 * len(ordered))))
    fit_rows = ordered[:-calibration_count]
    calibration_rows = ordered[-calibration_count:]
    model = ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=5,
        max_features=0.8,
        bootstrap=True,
        max_samples=0.85,
        random_state=seed,
        n_jobs=-1,
    ).fit(
        np.asarray([row.vector[start:end] for row in fit_rows], dtype=float),
        np.asarray([row.target for row in fit_rows], dtype=float),
    )
    calibration_x = np.asarray(
        [row.vector[start:end] for row in calibration_rows], dtype=float
    )
    calibration_y = np.asarray([row.target for row in calibration_rows], dtype=float)
    residual = float(
        math.sqrt(float(np.mean((model.predict(calibration_x) - calibration_y) ** 2)))
    )
    return _Regressor(model, max(1e-6, residual), (start, end))


def _predict_many(regressor, materials):
    start, end = regressor.feature_slice
    return [
        float(value)
        for value in regressor.model.predict([row.vector[start:end] for row in materials])
    ]


def _percentile(values, value):
    import bisect

    return (bisect.bisect_right(values, value) + 0.5) / (len(values) + 1.0)


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
    return max(1e-6, _quantile(values, 0.75) - _quantile(values, 0.25))


def _logistic(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(60.0, value)))
    exponential = math.exp(max(-60.0, value))
    return exponential / (1.0 + exponential)
