from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .external_h14 import ExternalMaterial, _load_snapshots, _quantile
from .schema import ActionRecord


EVIDENCE_SOURCES = ("composition", "structure", "high_fidelity")


@dataclass(frozen=True)
class _SourceRegressor:
    model: Any
    residual_scale: float
    feature_slice: tuple[int, int]


@dataclass(frozen=True)
class _SourceNovelty:
    scaler: Any
    neighbors: Any
    training_distances: tuple[float, ...]
    feature_slice: tuple[int, int]


@dataclass(frozen=True)
class _SourceEstimate:
    margin: float
    uncertainty: float
    utility: float
    utility_low: float
    utility_high: float
    details: dict[str, Any]


class ExternalH14EvidenceEnvironment:
    def __init__(
        self,
        materials_by_dataset: dict[str, list[ExternalMaterial]],
        regressors: dict[tuple[str, str], _SourceRegressor],
        novelty: dict[str, _SourceNovelty],
        source_thresholds: dict[str, float],
        extreme_threshold: float,
        target_values_by_task: dict[str, tuple[float, ...]],
    ) -> None:
        self.materials_by_dataset = materials_by_dataset
        self.material_by_dataset_and_id = {
            (dataset, material.row_id): material
            for dataset, materials in materials_by_dataset.items()
            for material in materials
        }
        self.regressors = regressors
        self.novelty = novelty
        self.source_thresholds = source_thresholds
        self.extreme_threshold = extreme_threshold
        self.target_values_by_task = target_values_by_task

    @classmethod
    def fit(
        cls,
        snapshot_root: str | Path,
        train_records: list[ActionRecord],
        *,
        seed: int,
    ) -> "ExternalH14EvidenceEnvironment":
        materials_by_dataset = _load_snapshots(snapshot_root)
        train_ids = _training_row_ids(train_records)
        specifications = {
            "external_pairwise_dielectric": "matbench_dielectric",
            "external_unique_jdft2d": "matbench_jdft2d",
            "external_extreme_phonons": "matbench_phonons",
        }
        regressors = {}
        for task_index, (task, dataset) in enumerate(specifications.items()):
            rows = [
                material
                for material in materials_by_dataset[dataset]
                if material.row_id in train_ids[dataset]
            ]
            for source_index, source in enumerate(("composition", "structure")):
                regressors[(task, source)] = _fit_source_regressor(
                    rows,
                    source=source,
                    seed=seed + task_index * 503 + source_index * 97,
                )
        unique_train = [
            material
            for material in materials_by_dataset["matbench_jdft2d"]
            if material.row_id in train_ids["matbench_jdft2d"]
        ]
        novelty = {
            source: _fit_source_novelty(unique_train, source=source)
            for source in ("composition", "structure")
        }
        novelty["high_fidelity"] = _fit_source_novelty(
            unique_train, source="combined"
        )
        unique_targets = sorted(material.target for material in unique_train)
        source_thresholds = {}
        for source in ("composition", "structure"):
            regressor = regressors[("external_unique_jdft2d", source)]
            scores = []
            for material in unique_train:
                prediction = _predict_source(regressor, material)
                performance = 1.0 - _empirical_percentile(unique_targets, prediction)
                uniqueness = _novelty_percentile(novelty[source], material, exclude_self=True)
                scores.append(0.5 * (performance + uniqueness))
            source_thresholds[source] = _quantile(scores, 0.90)
        exact_scores = [
            0.5 * (1.0 - _empirical_percentile(unique_targets, material.target))
            + 0.5 * _novelty_percentile(
                novelty["high_fidelity"], material, exclude_self=True
            )
            for material in unique_train
        ]
        source_thresholds["high_fidelity"] = _quantile(exact_scores, 0.90)
        extreme_train = [
            material.target
            for material in materials_by_dataset["matbench_phonons"]
            if material.row_id in train_ids["matbench_phonons"]
        ]
        return cls(
            materials_by_dataset,
            regressors,
            novelty,
            source_thresholds,
            _quantile(extreme_train, 0.90),
            {
                "external_pairwise_dielectric": tuple(
                    material.target
                    for material in materials_by_dataset["matbench_dielectric"]
                    if material.row_id in train_ids["matbench_dielectric"]
                ),
                "external_unique_jdft2d": tuple(unique_targets),
                "external_extreme_phonons": tuple(extreme_train),
            },
        )

    def evidence_views(self, record: ActionRecord) -> list[ActionRecord]:
        return [self.observe(record, source=source) for source in EVIDENCE_SOURCES]

    def observe(self, record: ActionRecord, *, source: str) -> ActionRecord:
        if source not in EVIDENCE_SOURCES:
            raise ValueError(f"unknown evidence source {source}")
        if record.benchmark == "external_pairwise_dielectric":
            estimate = self._pairwise(record, source)
        elif record.benchmark == "external_unique_jdft2d":
            estimate = self._unique(record, source)
        elif record.benchmark == "external_extreme_phonons":
            estimate = self._extreme(record, source)
        else:
            raise ValueError(f"unsupported external task {record.benchmark}")
        scale = max(1e-6, estimate.uncertainty + 1e-3)
        confidence = _stable_logistic(estimate.margin / scale)
        confidence_low = _stable_logistic(
            (estimate.margin - estimate.uncertainty) / scale
        )
        confidence_high = _stable_logistic(
            (estimate.margin + estimate.uncertainty) / scale
        )
        tool_cost = 0.020 if source == "high_fidelity" else 0.002
        return replace(
            record,
            evidence=[
                *record.evidence,
                f"{source} scientific evidence margin={estimate.margin:.6f}",
                f"{source} source residual scale={estimate.uncertainty:.6f}",
                f"{source} estimated scientific utility={estimate.utility:.6f}",
            ],
            features={
                **record.features,
                "tool_estimated_margin": estimate.margin,
                "tool_uncertainty": estimate.uncertainty,
                "tool_source_confidence": confidence,
                "tool_source_confidence_low": confidence_low,
                "tool_source_confidence_high": confidence_high,
                "tool_estimated_utility": estimate.utility,
                "tool_estimated_utility_low": estimate.utility_low,
                "tool_estimated_utility_high": estimate.utility_high,
                "tool_source_composition": float(source == "composition"),
                "tool_source_structure": float(source == "structure"),
                "tool_source_high_fidelity": float(source == "high_fidelity"),
                "tool_cost": tool_cost,
            },
            metadata={
                **record.metadata,
                "evidence_source": source,
                "evidence_provenance": estimate.details,
                "hidden_outcome_exposed": False,
                "outcome_defining_measurement_replayed": source == "high_fidelity",
            },
        )

    def _pairwise(self, record: ActionRecord, source: str):
        chosen = self.material_by_dataset_and_id[
            ("matbench_dielectric", int(record.metadata["chosen_row_id"]))
        ]
        alternative = self.material_by_dataset_and_id[
            ("matbench_dielectric", int(record.metadata["alternative_row_id"]))
        ]
        if source == "high_fidelity":
            scale = _robust_scale(self.target_values_by_task[record.benchmark])
            chosen_value = chosen.target
            alternative_value = alternative.target
            residual_scale = max(1e-6, 0.01 * scale)
            details = {
                "source_family": source,
                "tool": "costed_database_measurement_replay",
                "target_label_exposed": False,
            }
        else:
            regressor = self.regressors[(record.benchmark, source)]
            chosen_value = _predict_source(regressor, chosen)
            alternative_value = _predict_source(regressor, alternative)
            residual_scale = regressor.residual_scale
            details = {
                "source_family": source,
                "tool": "train_only_dielectric_property_model",
            }
        train_targets = list(self.target_values_by_task[record.benchmark])
        utility = _pair_utility(train_targets, chosen_value, alternative_value)
        utility_low = _pair_utility(
            train_targets,
            chosen_value - residual_scale,
            alternative_value + residual_scale,
        )
        utility_high = _pair_utility(
            train_targets,
            chosen_value + residual_scale,
            alternative_value - residual_scale,
        )
        return _SourceEstimate(
            margin=chosen_value - alternative_value,
            uncertainty=math.sqrt(2.0) * residual_scale,
            utility=utility,
            utility_low=min(utility, utility_low),
            utility_high=max(utility, utility_high),
            details=details,
        )

    def _unique(self, record: ActionRecord, source: str):
        material = self.material_by_dataset_and_id[
            ("matbench_jdft2d", int(record.metadata["source_row_id"]))
        ]
        train_targets = sorted(self.target_values_by_task[record.benchmark])
        if source == "high_fidelity":
            prediction = material.target
            residual_scale = max(1e-6, 0.01 * _robust_scale(train_targets))
        else:
            regressor = self.regressors[(record.benchmark, source)]
            prediction = _predict_source(regressor, material)
            residual_scale = regressor.residual_scale
        performance = 1.0 - _empirical_percentile(train_targets, prediction)
        uniqueness = _novelty_percentile(self.novelty[source], material, exclude_self=False)
        discovery = 0.5 * (performance + uniqueness)
        lower_performance = 1.0 - _empirical_percentile(
            train_targets, prediction + residual_scale
        )
        upper_performance = 1.0 - _empirical_percentile(
            train_targets, prediction - residual_scale
        )
        return _SourceEstimate(
            margin=discovery - self.source_thresholds[source],
            uncertainty=residual_scale,
            utility=discovery,
            utility_low=0.5 * (lower_performance + uniqueness),
            utility_high=0.5 * (upper_performance + uniqueness),
            details={
                "source_family": source,
                "tool": (
                    "costed_database_measurement_replay"
                    if source == "high_fidelity"
                    else "train_only_exfoliation_and_novelty_model"
                ),
                "target_label_exposed": False,
            },
        )

    def _extreme(self, record: ActionRecord, source: str):
        material = self.material_by_dataset_and_id[
            ("matbench_phonons", int(record.metadata["source_row_id"]))
        ]
        if source == "high_fidelity":
            prediction = material.target
            residual_scale = max(
                1e-6,
                0.01 * _robust_scale(self.target_values_by_task[record.benchmark]),
            )
        else:
            regressor = self.regressors[(record.benchmark, source)]
            prediction = _predict_source(regressor, material)
            residual_scale = regressor.residual_scale
        train_targets = list(self.target_values_by_task[record.benchmark])
        utility = _empirical_percentile(train_targets, prediction)
        return _SourceEstimate(
            margin=prediction - self.extreme_threshold,
            uncertainty=residual_scale,
            utility=utility,
            utility_low=_empirical_percentile(train_targets, prediction - residual_scale),
            utility_high=_empirical_percentile(train_targets, prediction + residual_scale),
            details={
                "source_family": source,
                "tool": (
                    "costed_database_measurement_replay"
                    if source == "high_fidelity"
                    else "train_only_phonon_property_model"
                ),
                "target_label_exposed": False,
            },
        )


def _training_row_ids(records: list[ActionRecord]) -> dict[str, set[int]]:
    output = {
        "matbench_dielectric": set(),
        "matbench_jdft2d": set(),
        "matbench_phonons": set(),
    }
    for record in records:
        dataset = str(record.metadata["source_dataset"])
        if dataset == "matbench_dielectric":
            output[dataset].add(int(record.metadata["chosen_row_id"]))
            output[dataset].add(int(record.metadata["alternative_row_id"]))
        else:
            output[dataset].add(int(record.metadata["source_row_id"]))
    return output


def _feature_slice(source: str) -> tuple[int, int]:
    if source == "composition":
        return 0, 119
    if source == "structure":
        return 118, 128
    if source == "combined":
        return 0, 128
    raise ValueError(source)


def _fit_source_regressor(
    materials: list[ExternalMaterial], *, source: str, seed: int
) -> _SourceRegressor:
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
    fitted = _SourceRegressor(model, max(1e-6, residual), (start, end))
    object.__setattr__(fitted, "_training_ids", frozenset(material.row_id for material in materials))
    return fitted


def _regressor_training_ids(regressor: _SourceRegressor) -> frozenset[int]:
    return getattr(regressor, "_training_ids", frozenset())


def _predict_source(regressor: _SourceRegressor, material: ExternalMaterial) -> float:
    start, end = regressor.feature_slice
    return float(regressor.model.predict([material.vector[start:end]])[0])


def _fit_source_novelty(
    materials: list[ExternalMaterial], *, source: str
) -> _SourceNovelty:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    start, end = _feature_slice(source)
    matrix = np.asarray([material.vector[start:end] for material in materials], dtype=float)
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    neighbors = NearestNeighbors(n_neighbors=2).fit(standardized)
    distances = neighbors.kneighbors(standardized)[0][:, 1]
    return _SourceNovelty(
        scaler,
        neighbors,
        tuple(sorted(float(value) for value in distances)),
        (start, end),
    )


def _novelty_percentile(
    novelty: _SourceNovelty, material: ExternalMaterial, *, exclude_self: bool
) -> float:
    start, end = novelty.feature_slice
    transformed = novelty.scaler.transform([material.vector[start:end]])
    count = 2 if exclude_self else 1
    distance = float(novelty.neighbors.kneighbors(transformed, n_neighbors=count)[0][0][-1])
    return _empirical_percentile(list(novelty.training_distances), distance)


def _empirical_percentile(values: list[float], value: float) -> float:
    import bisect

    ordered = values if values == sorted(values) else sorted(values)
    return (bisect.bisect_right(ordered, value) + 0.5) / (len(ordered) + 1.0)


def _pair_utility(values: list[float], chosen: float, alternative: float) -> float:
    return max(
        0.0,
        _empirical_percentile(values, chosen)
        - _empirical_percentile(values, alternative),
    )


def _stable_logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(60.0, value)))
    exponential = math.exp(max(-60.0, value))
    return exponential / (1.0 + exponential)


def _robust_scale(values: tuple[float, ...] | list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) < 4:
        return max(1e-6, max(ordered) - min(ordered))
    return max(1e-6, _quantile(ordered, 0.75) - _quantile(ordered, 0.25))
