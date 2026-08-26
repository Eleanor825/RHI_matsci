from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

from .hurdle_gain_calibration import HurdleGainPrediction
from .source_fusion import SourceFusionObservation
from .source_decomposition import decompose_source_variance


@dataclass(frozen=True)
class EvidenceValueObservation:
    record_id: str
    benchmark: str
    base_mean_gain: float
    base_p_positive_gain: float
    base_expected_shortfall: float
    base_tail_risk_score: float
    base_internal_variance: float
    base_external_variance: float
    tool_cost: float
    realized_value: float
    base_interaction_variance: float = 0.0
    base_source_range: float = 0.0
    base_source_variance: float = 0.0
    base_source_sign_disagreement: float = 0.0
    base_source_positive_fraction: float = 0.0
    base_internal_prior_mean: float = 0.0
    base_internal_prior_variance: float = 0.0
    base_external_mean: float = 0.0
    base_internal_external_gap: float = 0.0
    base_external_consensus_variance: float = 0.0
    base_source_interaction_variance: float = 0.0
    context_evidence_support: float = 0.0
    context_evidence_conflict: float = 0.0
    context_ood_score: float = 0.0
    context_source_reliability: float = 0.0
    context_perturbation_stability: float = 0.0
    candidate_tool_composition: float = 0.0
    candidate_tool_structure: float = 0.0
    candidate_tool_high_fidelity: float = 0.0


@dataclass(frozen=True)
class EvidenceValuePrediction:
    record_id: str
    benchmark: str
    expected_value: float
    internal_variance: float
    residual_variance: float
    predictive_variance: float
    positive_value_probability: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceValueModel:
    models_by_task: dict[str, tuple[object, ...]]
    classifiers_by_task: dict[str, tuple[object, ...]]
    residual_variance_by_task: dict[str, float]

    def predict_one(self, observation: EvidenceValueObservation) -> EvidenceValuePrediction:
        return self.predict([observation])[0]

    def predict(
        self, observations: Iterable[EvidenceValueObservation]
    ) -> list[EvidenceValuePrediction]:
        rows = list(observations)
        output: list[EvidenceValuePrediction | None] = [None] * len(rows)
        for task in sorted({row.benchmark for row in rows}):
            if task not in self.models_by_task:
                raise ValueError(f"no evidence-value model for task {task}")
            indices = [index for index, row in enumerate(rows) if row.benchmark == task]
            matrix = [_feature_row(rows[index]) for index in indices]
            regression = [model.predict(matrix) for model in self.models_by_task[task]]
            classification = [
                classifier.predict_proba(matrix)[:, 1]
                for classifier in self.classifiers_by_task[task]
            ]
            residual_variance = self.residual_variance_by_task[task]
            for column, index in enumerate(indices):
                replica_values = [float(values[column]) for values in regression]
                probabilities = [float(values[column]) for values in classification]
                expected_value = sum(replica_values) / len(replica_values)
                internal_variance = _population_variance(replica_values)
                predictive_variance = internal_variance + residual_variance
                output[index] = EvidenceValuePrediction(
                    record_id=rows[index].record_id,
                    benchmark=task,
                    expected_value=expected_value,
                    internal_variance=internal_variance,
                    residual_variance=residual_variance,
                    predictive_variance=predictive_variance,
                    positive_value_probability=min(
                        1.0 - 1e-9,
                        max(1e-9, sum(probabilities) / len(probabilities)),
                    ),
                )
        if any(row is None for row in output):
            raise RuntimeError("evidence-value batch prediction missed records")
        return [row for row in output if row is not None]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "counterfactual_evidence_value_ensemble",
            "replicas_by_task": {
                task: len(models) for task, models in sorted(self.models_by_task.items())
            },
            "classifier_replicas_by_task": {
                task: len(models)
                for task, models in sorted(self.classifiers_by_task.items())
            },
            "residual_variance_by_task": dict(sorted(self.residual_variance_by_task.items())),
            "features": list(_FEATURE_NAMES),
        }


def realized_counterfactual_evidence_value(
    base_prediction: HurdleGainPrediction,
    tool_prediction: HurdleGainPrediction,
    *,
    realized_gain: float,
    tool_cost: float,
    base_execution_threshold: float = 0.0,
    tool_execution_threshold: float = 0.0,
) -> float:
    if base_prediction.record_id != tool_prediction.record_id:
        raise ValueError("base and tool predictions must refer to the same record")
    base_execute = float(
        base_prediction.tail_risk_score >= base_execution_threshold
    )
    tool_execute = float(
        tool_prediction.tail_risk_score >= tool_execution_threshold
    )
    return realized_gain * (tool_execute - base_execute) - tool_cost


def evidence_value_observation(
    base_prediction: HurdleGainPrediction,
    tool_prediction: HurdleGainPrediction,
    *,
    realized_gain: float,
    tool_cost: float,
    source_observation: SourceFusionObservation | None = None,
    base_execution_threshold: float = 0.0,
    tool_execution_threshold: float = 0.0,
) -> EvidenceValueObservation:
    source_features = source_disagreement_features(source_observation)
    return EvidenceValueObservation(
        record_id=base_prediction.record_id,
        benchmark=base_prediction.benchmark,
        base_mean_gain=base_prediction.mean_gain,
        base_p_positive_gain=base_prediction.p_positive_gain,
        base_expected_shortfall=base_prediction.expected_shortfall,
        base_tail_risk_score=base_prediction.tail_risk_score,
        base_internal_variance=base_prediction.calibrated_internal_variance,
        base_external_variance=base_prediction.calibrated_external_variance,
        base_interaction_variance=base_prediction.calibrated_interaction_variance,
        tool_cost=tool_cost,
        realized_value=realized_counterfactual_evidence_value(
            base_prediction,
            tool_prediction,
            realized_gain=realized_gain,
            tool_cost=tool_cost,
            base_execution_threshold=base_execution_threshold,
            tool_execution_threshold=tool_execution_threshold,
        ),
        **source_features,
    )


def source_disagreement_features(
    observation: SourceFusionObservation | None,
) -> dict[str, float]:
    if observation is None:
        return {
            "base_source_range": 0.0,
            "base_source_variance": 0.0,
            "base_source_sign_disagreement": 0.0,
            "base_source_positive_fraction": 0.0,
            "base_internal_prior_mean": 0.0,
            "base_internal_prior_variance": 0.0,
            "base_external_mean": 0.0,
            "base_internal_external_gap": 0.0,
            "base_external_consensus_variance": 0.0,
            "base_source_interaction_variance": 0.0,
        }
    source_count = len(observation.source_names)
    means = [
        sum(row[source_index] for row in observation.predicted_gain)
        / len(observation.predicted_gain)
        for source_index in range(source_count)
    ]
    center = sum(means) / len(means)
    decomposition_interaction = (
        decompose_source_variance(
            [list(row) for row in observation.predicted_gain]
        ).interaction_variance
        if source_count >= 2
        else 0.0
    )
    internal_indices = [
        index
        for index, source in enumerate(observation.source_names)
        if source == "agent_prior" or source.startswith("agent_internal")
    ]
    external_indices = [
        index for index in range(source_count) if index not in internal_indices
    ]
    if internal_indices:
        internal_values = [means[index] for index in internal_indices]
        internal_prior_mean = sum(internal_values) / len(internal_values)
        internal_replica_values = [
            sum(row[index] for index in internal_indices) / len(internal_indices)
            for row in observation.predicted_gain
        ]
        internal_prior_variance = _population_variance(internal_replica_values)
    else:
        internal_prior_mean = 0.0
        internal_prior_variance = 0.0
    if external_indices:
        external_values = [means[index] for index in external_indices]
        external_mean = sum(external_values) / len(external_values)
        external_consensus_variance = _population_variance(external_values)
    else:
        external_mean = 0.0
        external_consensus_variance = 0.0
    return {
        "base_source_range": max(means) - min(means),
        "base_source_variance": sum((value - center) ** 2 for value in means)
        / len(means),
        "base_source_sign_disagreement": float(min(means) < 0.0 < max(means)),
        "base_source_positive_fraction": sum(value > 0.0 for value in means)
        / len(means),
        "base_internal_prior_mean": internal_prior_mean,
        "base_internal_prior_variance": internal_prior_variance,
        "base_external_mean": external_mean,
        "base_internal_external_gap": (
            internal_prior_mean - external_mean if internal_indices and external_indices else 0.0
        ),
        "base_external_consensus_variance": external_consensus_variance,
        "base_source_interaction_variance": decomposition_interaction,
    }


def fit_evidence_value_model(
    observations: list[EvidenceValueObservation],
    *,
    seed: int,
    replicas: int = 5,
    trees: int = 160,
    model_family: str = "extra_trees",
) -> EvidenceValueModel:
    if not observations:
        raise ValueError("evidence-value fitting requires observations")
    if replicas < 2:
        raise ValueError("at least two evidence-value replicas are required")
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models_by_task = {}
    classifiers_by_task = {}
    residual_variance_by_task = {}
    for task_index, task in enumerate(
        sorted({observation.benchmark for observation in observations})
    ):
        rows = [observation for observation in observations if observation.benchmark == task]
        if len(rows) < 10:
            raise ValueError(f"task {task} needs at least ten evidence-value observations")
        matrix = np.asarray([_feature_row(row) for row in rows], dtype=float)
        targets = np.asarray([row.realized_value for row in rows], dtype=float)
        positive_targets = np.asarray([row.realized_value > 0.0 for row in rows], dtype=int)
        if len(set(positive_targets.tolist())) < 2:
            raise ValueError(f"task {task} needs both positive and non-positive tool values")
        models = []
        classifiers = []
        for replica in range(replicas):
            replica_seed = seed * 1009 + task_index * 503 + replica * 97
            if model_family == "extra_trees":
                model = ExtraTreesRegressor(
                    n_estimators=trees,
                    min_samples_leaf=6,
                    max_features=0.8,
                    bootstrap=True,
                    max_samples=0.85,
                    random_state=replica_seed,
                    n_jobs=-1,
                )
                classifier = ExtraTreesClassifier(
                    n_estimators=trees,
                    min_samples_leaf=3,
                    max_features=0.8,
                    bootstrap=True,
                    max_samples=0.85,
                    class_weight="balanced_subsample",
                    random_state=replica_seed + 17,
                    n_jobs=-1,
                )
                fit_matrix = matrix
                fit_targets = targets
                fit_positive_targets = positive_targets
            elif model_family == "regularized_linear":
                generator = np.random.default_rng(replica_seed)
                indices = generator.choice(
                    len(rows), size=max(10, int(round(0.85 * len(rows)))), replace=True
                )
                fit_matrix = matrix[indices]
                fit_targets = targets[indices]
                fit_positive_targets = positive_targets[indices]
                model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.10,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=replica_seed + 17,
                    ),
                )
            else:
                raise ValueError(f"unknown evidence-value model family {model_family}")
            model.fit(fit_matrix, fit_targets)
            models.append(model)
            classifier.fit(fit_matrix, fit_positive_targets)
            classifiers.append(classifier)
        ensemble_mean = np.mean(
            np.asarray([model.predict(matrix) for model in models], dtype=float), axis=0
        )
        residual_variance = float(np.mean((targets - ensemble_mean) ** 2))
        models_by_task[task] = tuple(models)
        classifiers_by_task[task] = tuple(classifiers)
        residual_variance_by_task[task] = max(1e-10, residual_variance)
    return EvidenceValueModel(
        models_by_task,
        classifiers_by_task,
        residual_variance_by_task,
    )


def cross_fitted_evidence_value_predictions(
    observations: list[EvidenceValueObservation],
    *,
    seed: int,
    folds: int = 5,
    replicas: int = 5,
    trees: int = 160,
    model_family: str = "extra_trees",
) -> list[EvidenceValuePrediction]:
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    assignments = [
        _fold_assignment(row.record_id, row.benchmark, folds=folds, seed=seed)
        for row in observations
    ]
    output: list[EvidenceValuePrediction | None] = [None] * len(observations)
    for fold in range(folds):
        train = [row for row, assignment in zip(observations, assignments) if assignment != fold]
        held_indices = [
            index for index, assignment in enumerate(assignments) if assignment == fold
        ]
        if not held_indices:
            continue
        model = fit_evidence_value_model(
            train,
            seed=seed + fold * 1009,
            replicas=replicas,
            trees=trees,
            model_family=model_family,
        )
        predictions = model.predict([observations[index] for index in held_indices])
        for index, prediction in zip(held_indices, predictions):
            output[index] = prediction
    if any(prediction is None for prediction in output):
        raise RuntimeError("cross-fitting did not predict every evidence-value observation")
    return [prediction for prediction in output if prediction is not None]


_FEATURE_NAMES = (
    "base_mean_gain",
    "base_p_positive_gain",
    "base_expected_shortfall",
    "base_tail_risk_score",
    "base_internal_variance",
    "base_external_variance",
    "base_interaction_variance",
    "tool_cost",
    "base_source_range",
    "base_source_variance",
    "base_source_sign_disagreement",
    "base_source_positive_fraction",
    "base_internal_prior_mean",
    "base_internal_prior_variance",
    "base_external_mean",
    "base_internal_external_gap",
    "base_external_consensus_variance",
    "base_source_interaction_variance",
    "context_evidence_support",
    "context_evidence_conflict",
    "context_ood_score",
    "context_source_reliability",
    "context_perturbation_stability",
    "candidate_tool_composition",
    "candidate_tool_structure",
    "candidate_tool_high_fidelity",
)


def _feature_row(observation: EvidenceValueObservation) -> list[float]:
    return [float(getattr(observation, name)) for name in _FEATURE_NAMES]


def _population_variance(values: list[float]) -> float:
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _fold_assignment(record_id: str, benchmark: str, *, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{benchmark}:{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds
