from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression

from .features import feature_names_from_records
from .replica_contract import EvidenceView
from .schema import ActionRecord
from .scientific_gain import GainTarget


@dataclass(frozen=True)
class CandidateSimulation:
    calibrated_positive_gain_probability: float
    mean_predicted_gain: float
    probability_disagreement: float
    gain_disagreement: float
    model_probabilities: tuple[float, ...]
    model_gains: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "calibrated_positive_gain_probability": self.calibrated_positive_gain_probability,
            "mean_predicted_gain": self.mean_predicted_gain,
            "probability_disagreement": self.probability_disagreement,
            "gain_disagreement": self.gain_disagreement,
            "model_probabilities": list(self.model_probabilities),
            "model_gains": list(self.model_gains),
        }


class CandidateSimulatorTool:
    """Train-only candidate-specific surrogate ensemble used as an auditable tool."""

    def __init__(
        self,
        train_records: list[ActionRecord],
        train_targets: list[GainTarget],
        feedback_records: list[ActionRecord],
        feedback_targets: list[GainTarget],
        *,
        seed: int = 260712447,
    ) -> None:
        if len(train_records) != len(train_targets):
            raise ValueError("train records and targets must align")
        if len(feedback_records) != len(feedback_targets):
            raise ValueError("feedback records and targets must align")
        self.feature_names = feature_names_from_records(train_records)
        self.models_by_task: dict[str, dict[str, Any]] = {}
        tasks = sorted({record.benchmark for record in train_records})
        for task_index, task in enumerate(tasks):
            train_indices = [index for index, record in enumerate(train_records) if record.benchmark == task]
            feedback_indices = [index for index, record in enumerate(feedback_records) if record.benchmark == task]
            x_train = self._matrix([train_records[index] for index in train_indices])
            y_train = np.asarray([train_targets[index].worthy for index in train_indices], dtype=int)
            g_train = np.asarray([train_targets[index].gain for index in train_indices], dtype=float)
            x_feedback = self._matrix([feedback_records[index] for index in feedback_indices])
            y_feedback = np.asarray([feedback_targets[index].worthy for index in feedback_indices], dtype=int)
            task_seed = seed + 1000 * task_index
            classifiers = (
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=task_seed + 1,
                    n_jobs=-1,
                ),
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=task_seed + 2,
                    n_jobs=-1,
                ),
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=task_seed + 3,
                ),
            )
            regressors = (
                ExtraTreesRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=task_seed + 11,
                    n_jobs=-1,
                ),
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=task_seed + 12,
                    n_jobs=-1,
                ),
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=task_seed + 13,
                ),
            )
            for model in (*classifiers, *regressors):
                model.fit(x_train, y_train if hasattr(model, "predict_proba") else g_train)
            feedback_raw = np.mean(
                np.column_stack([model.predict_proba(x_feedback)[:, 1] for model in classifiers]),
                axis=1,
            )
            calibrator = _fit_platt(feedback_raw, y_feedback)
            self.models_by_task[task] = {
                "classifiers": classifiers,
                "regressors": regressors,
                "calibrator": calibrator,
            }

    def simulate(self, record: ActionRecord) -> CandidateSimulation:
        if record.benchmark not in self.models_by_task:
            raise ValueError(f"no simulator for task {record.benchmark}")
        models = self.models_by_task[record.benchmark]
        row = self._matrix([record])
        probabilities = tuple(
            float(model.predict_proba(row)[0, 1]) for model in models["classifiers"]
        )
        gains = tuple(float(model.predict(row)[0]) for model in models["regressors"])
        calibrated = _apply_platt(models["calibrator"], mean(probabilities))
        return CandidateSimulation(
            calibrated_positive_gain_probability=calibrated,
            mean_predicted_gain=mean(gains),
            probability_disagreement=pstdev(probabilities),
            gain_disagreement=pstdev(gains),
            model_probabilities=probabilities,
            model_gains=gains,
        )

    def evidence_view(self, record: ActionRecord) -> EvidenceView:
        simulation = self.simulate(record)
        agreement = max(0.0, 1.0 - min(1.0, 4.0 * simulation.probability_disagreement))
        features = dict(record.features)
        features.update(
            {
                "evidence_support": simulation.calibrated_positive_gain_probability,
                "evidence_conflict": 1.0 - agreement,
                "tool_agreement": agreement,
                "source_reliability": 0.9,
                "source_risk": 0.1,
                "candidate_simulator_probability": simulation.calibrated_positive_gain_probability,
                "candidate_simulator_gain": simulation.mean_predicted_gain,
                "candidate_simulator_disagreement": simulation.probability_disagreement,
            }
        )
        evidence = [
            *record.evidence,
            "Train-only candidate-specific simulator ensemble (target outcome excluded):",
            f"- calibrated_probability_positive_net_gain={simulation.calibrated_positive_gain_probability:.6f}",
            f"- mean_predicted_budget_conditioned_gain={simulation.mean_predicted_gain:.6f}",
            f"- classifier_probability_disagreement={simulation.probability_disagreement:.6f}",
            f"- regressor_gain_disagreement={simulation.gain_disagreement:.6f}",
        ]
        return EvidenceView(
            view_id="tool::candidate_simulator",
            record=replace(record, evidence=evidence, features=features),
            intervention="train_only_candidate_specific_surrogate_ensemble",
            provenance={
                "type": "offline_candidate_simulation_tool",
                "target_outcome_excluded": True,
                "training_scope": "train_only_models_feedback_only_calibration",
                "simulation": simulation.to_json(),
            },
        )

    def _matrix(self, records: list[ActionRecord]) -> np.ndarray:
        return np.asarray(
            [[float(record.features.get(name, 0.0)) for name in self.feature_names] for record in records],
            dtype=float,
        )


def _fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression | None:
    if len(set(int(value) for value in labels)) < 2:
        return None
    logits = np.asarray([_logit(float(value)) for value in probabilities]).reshape(-1, 1)
    return LogisticRegression(C=1e6, solver="lbfgs").fit(logits, labels)


def _apply_platt(calibrator: LogisticRegression | None, probability: float) -> float:
    if calibrator is None:
        return probability
    return float(calibrator.predict_proba(np.asarray([[_logit(probability)]]))[0, 1])


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(clipped / (1.0 - clipped))
