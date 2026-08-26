from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .adaptive_gain_harness import crossed_numeric_estimate


@dataclass(frozen=True)
class FactorizedGainTrainingRow:
    record_id: str
    benchmark: str
    features: Mapping[str, float]
    success: int
    normalized_utility: float
    normalized_action_cost: float
    failure_harm: float
    cumulative_evidence_cost: float = 0.0


@dataclass(frozen=True)
class CrossedFactorizedPrediction:
    record_id: str
    benchmark: str
    success_probability: float
    conditional_utility: float
    expected_net_gain: float
    p_positive_gain: float
    gain_internal_variance: float
    gain_external_variance: float
    gain_interaction_variance: float
    aleatoric_gain_variance: float
    predictive_gain_variance: float
    gain_cells: tuple[tuple[float, ...], ...]
    success_cells: tuple[tuple[float, ...], ...]
    utility_cells: tuple[tuple[float, ...], ...]

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("gain_cells")
        payload.pop("success_cells")
        payload.pop("utility_cells")
        return payload


@dataclass(frozen=True)
class _Replica:
    success_model: object
    utility_model: object


@dataclass(frozen=True)
class AdaptiveFactorizedGainModel:
    feature_names_by_task: dict[str, tuple[str, ...]]
    replicas_by_task: dict[str, tuple[_Replica, ...]]
    utility_residual_variance_by_task: dict[str, float]
    cost_weight: float = 0.15
    failure_harm_weight: float = 0.25

    def predict_crossed(
        self,
        *,
        record_id: str,
        benchmark: str,
        feature_views: Sequence[Mapping[str, float]],
        normalized_action_cost: float,
        failure_harm: float,
        cumulative_evidence_cost: float,
    ) -> CrossedFactorizedPrediction:
        if benchmark not in self.replicas_by_task:
            raise ValueError(f"no factorized gain model for task {benchmark}")
        if len(feature_views) < 2:
            raise ValueError("crossed prediction requires at least two evidence views")
        names = self.feature_names_by_task[benchmark]
        matrix = [_vectorize(view, names) for view in feature_views]
        success_cells = []
        utility_cells = []
        gain_cells = []
        charged_cost = self.cost_weight * (
            max(0.0, float(normalized_action_cost))
            + max(0.0, float(cumulative_evidence_cost))
        )
        harm = max(0.0, float(failure_harm))
        for replica in self.replicas_by_task[benchmark]:
            probabilities = _positive_probability(replica.success_model, matrix)
            utilities = [
                min(1.0, max(0.0, float(value)))
                for value in replica.utility_model.predict(matrix)
            ]
            gains = [
                probability * utility
                - charged_cost
                - self.failure_harm_weight * (1.0 - probability) * harm
                for probability, utility in zip(probabilities, utilities)
            ]
            success_cells.append(probabilities)
            utility_cells.append(utilities)
            gain_cells.append(gains)
        utility_residual = self.utility_residual_variance_by_task[benchmark]
        flattened_success = [value for row in success_cells for value in row]
        flattened_utility = [value for row in utility_cells for value in row]
        mean_success = sum(flattened_success) / len(flattened_success)
        mean_utility = sum(flattened_utility) / len(flattened_utility)
        aleatoric = (
            mean_success * utility_residual
            + mean_success
            * (1.0 - mean_success)
            * (mean_utility + self.failure_harm_weight * harm) ** 2
        )
        estimate = crossed_numeric_estimate(
            gain_cells,
            aleatoric_variance=aleatoric,
        )
        return CrossedFactorizedPrediction(
            record_id=record_id,
            benchmark=benchmark,
            success_probability=mean_success,
            conditional_utility=mean_utility,
            expected_net_gain=estimate.mean,
            p_positive_gain=estimate.positive_probability,
            gain_internal_variance=estimate.internal_variance,
            gain_external_variance=estimate.external_variance,
            gain_interaction_variance=estimate.interaction_variance,
            aleatoric_gain_variance=estimate.aleatoric_variance,
            predictive_gain_variance=estimate.predictive_variance,
            gain_cells=tuple(tuple(row) for row in gain_cells),
            success_cells=tuple(tuple(row) for row in success_cells),
            utility_cells=tuple(tuple(row) for row in utility_cells),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "type": "adaptive_factorized_gain_bootstrap_ensemble",
            "features_by_task": {
                task: list(names)
                for task, names in sorted(self.feature_names_by_task.items())
            },
            "replicas_by_task": {
                task: len(replicas)
                for task, replicas in sorted(self.replicas_by_task.items())
            },
            "utility_residual_variance_by_task": dict(
                sorted(self.utility_residual_variance_by_task.items())
            ),
            "cost_weight": self.cost_weight,
            "failure_harm_weight": self.failure_harm_weight,
        }


def fit_adaptive_factorized_gain_model(
    rows: Iterable[FactorizedGainTrainingRow],
    *,
    seed: int,
    replicas: int = 5,
    trees: int = 160,
    min_samples_leaf: int = 5,
    cost_weight: float = 0.15,
    failure_harm_weight: float = 0.25,
) -> AdaptiveFactorizedGainModel:
    observations = list(rows)
    if not observations:
        raise ValueError("factorized gain fitting requires observations")
    if replicas < 2:
        raise ValueError("at least two model replicas are required")
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    feature_names_by_task = {}
    replicas_by_task = {}
    residuals_by_task = {}
    for task_index, task in enumerate(sorted({row.benchmark for row in observations})):
        task_rows = [row for row in observations if row.benchmark == task]
        if len(task_rows) < 20:
            raise ValueError(f"task {task} needs at least twenty training rows")
        labels = np.asarray([int(row.success) for row in task_rows], dtype=int)
        if len(set(labels.tolist())) < 2:
            raise ValueError(f"task {task} needs successful and failed actions")
        feature_names = tuple(
            sorted({name for row in task_rows for name in row.features})
        )
        matrix = np.asarray(
            [_vectorize(row.features, feature_names) for row in task_rows],
            dtype=float,
        )
        utilities = np.asarray(
            [float(row.normalized_utility) for row in task_rows], dtype=float
        )
        successful_indices = np.flatnonzero(labels == 1)
        if len(successful_indices) < 10:
            raise ValueError(f"task {task} needs at least ten successful actions")
        task_replicas = []
        for replica_index in range(replicas):
            replica_seed = seed * 1009 + task_index * 503 + replica_index * 97
            generator = np.random.default_rng(replica_seed)
            sampled = generator.choice(
                len(task_rows), size=len(task_rows), replace=True
            )
            sampled_successes = sampled[labels[sampled] == 1]
            if len(sampled_successes) < 5:
                sampled_successes = successful_indices
            classifier = ExtraTreesClassifier(
                n_estimators=trees,
                min_samples_leaf=min_samples_leaf,
                max_features=0.8,
                class_weight="balanced_subsample",
                random_state=replica_seed,
                n_jobs=1,
            )
            utility_model = ExtraTreesRegressor(
                n_estimators=trees,
                min_samples_leaf=min_samples_leaf,
                max_features=0.8,
                random_state=replica_seed + 17,
                n_jobs=1,
            )
            classifier.fit(matrix[sampled], labels[sampled])
            utility_model.fit(matrix[sampled_successes], utilities[sampled_successes])
            task_replicas.append(_Replica(classifier, utility_model))
        successful_matrix = matrix[successful_indices]
        utility_predictions = np.mean(
            np.asarray(
                [
                    replica.utility_model.predict(successful_matrix)
                    for replica in task_replicas
                ],
                dtype=float,
            ),
            axis=0,
        )
        residuals_by_task[task] = max(
            1e-10,
            float(
                np.mean(
                    (utilities[successful_indices] - utility_predictions) ** 2
                )
            ),
        )
        feature_names_by_task[task] = feature_names
        replicas_by_task[task] = tuple(task_replicas)
    return AdaptiveFactorizedGainModel(
        feature_names_by_task=feature_names_by_task,
        replicas_by_task=replicas_by_task,
        utility_residual_variance_by_task=residuals_by_task,
        cost_weight=cost_weight,
        failure_harm_weight=failure_harm_weight,
    )


def deterministic_group_fold(
    record_id: str,
    benchmark: str,
    group_id: str,
    *,
    folds: int,
    seed: int,
) -> int:
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    digest = hashlib.sha256(
        f"{seed}|adaptive-factorized|{benchmark}|{group_id}|{record_id}".encode()
    ).hexdigest()
    return int(digest[:16], 16) % folds


def _vectorize(features: Mapping[str, float], names: Sequence[str]) -> list[float]:
    values = []
    for name in names:
        value = float(features.get(name, 0.0))
        values.append(value if math.isfinite(value) else 0.0)
    return values


def _positive_probability(model: object, matrix: Sequence[Sequence[float]]) -> list[float]:
    probabilities = model.predict_proba(matrix)
    classes = [int(value) for value in model.classes_]
    positive_index = classes.index(1)
    return [
        min(1.0 - 1e-9, max(1e-9, float(row[positive_index])))
        for row in probabilities
    ]
