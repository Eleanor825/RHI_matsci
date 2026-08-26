from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

from .evidence_value import EvidenceValueObservation, _feature_row


@dataclass(frozen=True)
class TailVoIPrediction:
    record_id: str
    benchmark: str
    positive_probability: float
    positive_value_mean: float
    downside_magnitude_mean: float
    hurdle_expected_value: float
    downside_expectation: float
    epistemic_variance: float
    tail_score: float

    def to_json(self):
        return asdict(self)


@dataclass(frozen=True)
class TailVoIModel:
    classifiers_by_task: dict[str, tuple[object, ...]]
    positive_models_by_task: dict[str, tuple[object, ...]]
    downside_models_by_task: dict[str, tuple[object, ...]]
    tail_weight: float
    epistemic_weight: float

    def predict(self, observations):
        rows = list(observations)
        output = []
        for row in rows:
            matrix = [_feature_row(row)]
            classifiers = self.classifiers_by_task[row.benchmark]
            positive_models = self.positive_models_by_task[row.benchmark]
            downside_models = self.downside_models_by_task[row.benchmark]
            replicas = []
            probabilities = []
            positive_values = []
            downside_values = []
            for classifier, positive_model, downside_model in zip(
                classifiers, positive_models, downside_models
            ):
                probability = float(classifier.predict_proba(matrix)[0, 1])
                positive = max(0.0, float(positive_model.predict(matrix)[0]))
                downside = max(0.0, float(downside_model.predict(matrix)[0]))
                probabilities.append(probability)
                positive_values.append(positive)
                downside_values.append(downside)
                replicas.append(probability * positive - (1.0 - probability) * downside)
            probability = sum(probabilities) / len(probabilities)
            positive = sum(positive_values) / len(positive_values)
            downside = sum(downside_values) / len(downside_values)
            expected = probability * positive - (1.0 - probability) * downside
            downside_expectation = (1.0 - probability) * downside
            epistemic_variance = _variance(replicas)
            output.append(
                TailVoIPrediction(
                    record_id=row.record_id,
                    benchmark=row.benchmark,
                    positive_probability=probability,
                    positive_value_mean=positive,
                    downside_magnitude_mean=downside,
                    hurdle_expected_value=expected,
                    downside_expectation=downside_expectation,
                    epistemic_variance=epistemic_variance,
                    tail_score=(
                        expected
                        - self.tail_weight * downside_expectation
                        - self.epistemic_weight * math.sqrt(epistemic_variance)
                    ),
                )
            )
        return output

    def predict_one(self, observation):
        return self.predict([observation])[0]


def fit_tail_voi_model(
    observations: list[EvidenceValueObservation],
    *,
    seed: int,
    replicas: int = 5,
    tail_weight: float = 1.0,
    epistemic_weight: float = 0.5,
):
    if not observations:
        raise ValueError("tail VoE fitting requires observations")
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classifiers_by_task = {}
    positive_models_by_task = {}
    downside_models_by_task = {}
    for task_index, task in enumerate(sorted({row.benchmark for row in observations})):
        rows = [row for row in observations if row.benchmark == task]
        matrix = np.asarray([_feature_row(row) for row in rows], dtype=float)
        values = np.asarray([row.realized_value for row in rows], dtype=float)
        positive_labels = (values > 0.0).astype(int)
        if len(set(positive_labels.tolist())) < 2:
            raise ValueError(f"task {task} needs positive and non-positive VoE examples")
        positive_indices = np.flatnonzero(values > 0.0)
        downside_indices = np.flatnonzero(values <= 0.0)
        if len(positive_indices) < 5 or len(downside_indices) < 5:
            raise ValueError(f"task {task} has insufficient hurdle components")
        classifiers = []
        positive_models = []
        downside_models = []
        for replica in range(replicas):
            replica_seed = seed * 1009 + task_index * 503 + replica * 97
            generator = np.random.default_rng(replica_seed)
            all_bootstrap = generator.choice(
                len(rows), size=max(10, int(round(0.85 * len(rows)))), replace=True
            )
            positive_bootstrap = generator.choice(
                positive_indices,
                size=max(5, int(round(0.85 * len(positive_indices)))),
                replace=True,
            )
            downside_bootstrap = generator.choice(
                downside_indices,
                size=max(5, int(round(0.85 * len(downside_indices)))),
                replace=True,
            )
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.10,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=replica_seed,
                ),
            ).fit(matrix[all_bootstrap], positive_labels[all_bootstrap])
            positive_model = make_pipeline(
                StandardScaler(), Ridge(alpha=10.0)
            ).fit(matrix[positive_bootstrap], values[positive_bootstrap])
            downside_model = make_pipeline(
                StandardScaler(), Ridge(alpha=10.0)
            ).fit(matrix[downside_bootstrap], -values[downside_bootstrap])
            classifiers.append(classifier)
            positive_models.append(positive_model)
            downside_models.append(downside_model)
        classifiers_by_task[task] = tuple(classifiers)
        positive_models_by_task[task] = tuple(positive_models)
        downside_models_by_task[task] = tuple(downside_models)
    return TailVoIModel(
        classifiers_by_task,
        positive_models_by_task,
        downside_models_by_task,
        tail_weight,
        epistemic_weight,
    )


def cross_fitted_tail_voi_predictions(
    observations: list[EvidenceValueObservation],
    *,
    seed: int,
    folds: int = 5,
    replicas: int = 3,
    tail_weight: float = 1.0,
    epistemic_weight: float = 0.5,
):
    assignments = [
        _fold(row.record_id, row.benchmark, seed=seed, folds=folds)
        for row in observations
    ]
    output = [None] * len(observations)
    for fold in range(folds):
        train = [row for row, assignment in zip(observations, assignments) if assignment != fold]
        held = [index for index, assignment in enumerate(assignments) if assignment == fold]
        if not held:
            continue
        model = fit_tail_voi_model(
            train,
            seed=seed + fold * 1013,
            replicas=replicas,
            tail_weight=tail_weight,
            epistemic_weight=epistemic_weight,
        )
        predictions = model.predict([observations[index] for index in held])
        for index, prediction in zip(held, predictions):
            output[index] = prediction
    if any(row is None for row in output):
        raise RuntimeError("tail VoE cross-fitting missed observations")
    return [row for row in output if row is not None]


def _fold(record_id, benchmark, *, seed, folds):
    digest = hashlib.sha256(f"{seed}:{benchmark}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _variance(values):
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / len(values)
