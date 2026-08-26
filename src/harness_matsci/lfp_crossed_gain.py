from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, replace

from .conformal_gain import ConformalGainObservation, fit_split_conformal_gain
from .continuous_gain_calibration import (
    ContinuousGainObservation,
    fit_continuous_gain_calibrator,
)
from .metrics import binary_metrics
from .risk_control import binomial_upper_confidence
from .schema import ActionRecord
from .source_decomposition import decompose_source_variance
from .source_fusion import (
    SourceFusionObservation,
    SourceReliabilityFuser,
    fit_source_reliability_fuser,
)


SOLVENTS_AND_SALTS = frozenset(
    {"EC", "DMC", "EMC", "EA", "MA", "PC", "ACN", "戊腈", "LiPF6", "LiFSI", "LiFSO3"}
)
ADDITIVES = frozenset(
    {"DTD", "FEC", "VC", "TMSP", "ES", "MMDS", "LiPO2F2", "LiODFP", "LiODFB", "EC2DTD", "BiDTD"}
)
VIEW_NAMES = ("full", "solvent_salt", "additive", "historical_analog")
METHOD_COSTS = {"static": 0.0, "ensemble": 0.002, "crossed": 0.005}


@dataclass(frozen=True)
class RawGainState:
    record_id: str
    benchmark: str
    objective: str
    actual_comparative_gain: float
    proposal_sign: float
    static_predicted_gain: float
    ensemble_mean_gain: float
    ensemble_internal_variance: float
    crossed_mean_gain: float
    crossed_internal_variance: float
    crossed_external_variance: float
    crossed_interaction_variance: float
    source_matrix: tuple[tuple[float, ...], ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class CrossedPairwiseGainModel:
    def __init__(self, *, replicas: int = 12, ridge_alpha: float = 1.0, neighbors: int = 7):
        if replicas < 2:
            raise ValueError("at least two bootstrap replicas are required")
        self.replicas = replicas
        self.ridge_alpha = ridge_alpha
        self.neighbors = neighbors
        self.models_by_task = {}

    def fit(self, records: list[ActionRecord]) -> "CrossedPairwiseGainModel":
        by_task = {}
        for task in sorted({record.benchmark for record in records}):
            rows = [record for record in records if record.benchmark == task]
            by_task[task] = self._fit_task(rows, task)
        self.models_by_task = by_task
        return self

    def predict_one(self, record: ActionRecord) -> RawGainState:
        task_model = self.models_by_task[record.benchmark]
        static_difference = _predict_linear(task_model["static"], record)
        proposal_sign = 1.0 if static_difference >= 0.0 else -1.0
        matrix = []
        for replica in task_model["replicas"]:
            row = [
                proposal_sign * _predict_linear(replica["full"], record),
                proposal_sign * _predict_linear(replica["solvent_salt"], record),
                proposal_sign * _predict_linear(replica["additive"], record),
                proposal_sign * _predict_knn(replica["historical_analog"], record),
            ]
            matrix.append(row)
        source_variance = decompose_source_variance(matrix)
        full_values = [row[0] for row in matrix]
        hidden = record.metadata["hidden_outcome_for_evaluation_only"]
        actual_difference = float(hidden["utility_difference"])
        return RawGainState(
            record_id=record.record_id,
            benchmark=record.benchmark,
            objective=str(record.metadata["objective"]),
            actual_comparative_gain=proposal_sign * actual_difference,
            proposal_sign=proposal_sign,
            static_predicted_gain=proposal_sign * static_difference,
            ensemble_mean_gain=_mean(full_values),
            ensemble_internal_variance=_variance(full_values),
            crossed_mean_gain=_mean(value for row in matrix for value in row),
            crossed_internal_variance=source_variance.internal_variance,
            crossed_external_variance=source_variance.external_variance,
            crossed_interaction_variance=source_variance.interaction_variance,
            source_matrix=tuple(tuple(value for value in row) for row in matrix),
        )

    def predict(self, records: list[ActionRecord]) -> list[RawGainState]:
        return [self.predict_one(record) for record in records]

    def _fit_task(self, records: list[ActionRecord], task: str):
        if len(records) < 6:
            raise ValueError(f"task {task} has too few training records")
        feature_names = _view_feature_names(records)
        static = _fit_linear(records, feature_names["full"], self.ridge_alpha)
        replicas = []
        for replica_id in range(self.replicas):
            seed = int(hashlib.sha256(f"{task}:{replica_id}".encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            sample = [records[rng.randrange(len(records))] for _ in records]
            replicas.append(
                {
                    "full": _fit_linear(sample, feature_names["full"], self.ridge_alpha),
                    "solvent_salt": _fit_linear(
                        sample, feature_names["solvent_salt"], self.ridge_alpha
                    ),
                    "additive": _fit_linear(
                        sample, feature_names["additive"], self.ridge_alpha
                    ),
                    "historical_analog": _fit_knn(
                        sample,
                        feature_names["full"],
                        neighbors=self.neighbors,
                    ),
                }
            )
        return {"static": static, "replicas": replicas, "feature_names": feature_names}


def fit_candidate_excluded_source_fuser(
    records: list[ActionRecord],
    *,
    replicas: int = 6,
    regularization: float = 0.05,
) -> SourceReliabilityFuser:
    observations = []
    by_task = {
        task: [record for record in records if record.benchmark == task]
        for task in sorted({record.benchmark for record in records})
    }
    for task, task_records in by_task.items():
        for record in task_records:
            excluded = set(record.metadata["candidate_group_ids"])
            reduced = [
                candidate
                for candidate in task_records
                if excluded.isdisjoint(candidate.metadata["candidate_group_ids"])
            ]
            model = CrossedPairwiseGainModel(
                replicas=replicas,
                neighbors=min(3, len(reduced)),
            ).fit(reduced)
            state = model.predict_one(record)
            observations.append(
                SourceFusionObservation(
                    record_id=record.record_id,
                    benchmark="lfp_real_multitask_shared_source_fusion",
                    source_names=VIEW_NAMES,
                    predicted_gain=state.source_matrix,
                    gain=state.actual_comparative_gain,
                )
            )
    return fit_source_reliability_fuser(
        observations,
        regularization=regularization,
        fit_affine=False,
    )


def apply_source_fuser(
    states: list[RawGainState],
    fuser: SourceReliabilityFuser,
) -> list[RawGainState]:
    output = []
    for state in states:
        prediction = fuser.predict_one(
            SourceFusionObservation(
                record_id=state.record_id,
                benchmark="lfp_real_multitask_shared_source_fusion",
                source_names=VIEW_NAMES,
                predicted_gain=state.source_matrix,
                gain=state.actual_comparative_gain,
            )
        )
        output.append(
            replace(
                state,
                crossed_mean_gain=prediction.mean_gain,
                crossed_internal_variance=prediction.internal_variance,
                crossed_external_variance=prediction.external_variance,
                crossed_interaction_variance=prediction.interaction_variance,
            )
        )
    return output


def evaluate_calibrated_methods(
    *,
    feedback_states: list[RawGainState],
    acceptance_states: list[RawGainState],
    test_states: list[RawGainState],
    alpha: float = 0.10,
    delta: float = 0.05,
) -> dict[str, object]:
    output = {}
    calibration_states, certificate_states = _split_acceptance(acceptance_states)
    for method in ("static", "ensemble", "crossed"):
        feedback = [_observation(state, method) for state in feedback_states]
        calibrator = fit_continuous_gain_calibrator(
            feedback,
            use_source_variance=method != "static",
        )
        calibration_observations = [_observation(state, method) for state in calibration_states]
        calibration_predictions = calibrator.predict(calibration_observations)
        conformal = fit_split_conformal_gain(
            [
                ConformalGainObservation(
                    record_id=state.record_id,
                    predicted_mean_gain=prediction.calibrated_mean_gain,
                    predicted_scale=math.sqrt(prediction.calibrated_variance),
                    observed_gain=_net_gain(state, method),
                )
                for state, prediction in zip(calibration_states, calibration_predictions)
            ],
            alpha=alpha,
        )
        certificate = _predict_rows(
            certificate_states,
            method=method,
            calibrator=calibrator,
            conformal=conformal,
        )
        certificate_selected = [
            row for row in certificate if row["lower_gain_bound"] > 0.0
        ]
        certificate_errors = sum(
            int(row["net_gain"] <= 0.0) for row in certificate_selected
        )
        certificate_upper_bound = binomial_upper_confidence(
            certificate_errors,
            len(certificate_selected),
            delta,
        )
        certified = bool(certificate_selected) and certificate_upper_bound <= alpha
        test = _predict_rows(
            test_states,
            method=method,
            calibrator=calibrator,
            conformal=conformal,
        )
        selected = (
            [row for row in test if row["lower_gain_bound"] > 0.0]
            if certified
            else []
        )
        labels = [int(row["net_gain"] > 0.0) for row in test]
        probabilities = [row["p_positive_gain"] for row in test]
        metrics = binary_metrics(labels, probabilities, threshold=0.5)
        output[method] = {
            "cost": METHOD_COSTS[method],
            "calibrator": calibrator.to_json(),
            "conformal": conformal.to_json(),
            "certificate": {
                "policy": "lower_gain_bound > 0",
                "selected_count": len(certificate_selected),
                "selected_fraction": (
                    len(certificate_selected) / len(certificate) if certificate else 0.0
                ),
                "errors": certificate_errors,
                "empirical_risk": (
                    certificate_errors / len(certificate_selected)
                    if certificate_selected
                    else 0.0
                ),
                "risk_upper_bound": certificate_upper_bound,
                "alpha": alpha,
                "delta": delta,
                "certified": certified,
            },
            "test_metrics": metrics,
            "test_conformal_coverage": _mean(
                float(row["net_gain"] >= row["lower_gain_bound"]) for row in test
            ),
            "deployed": {
                "selected_count": len(selected),
                "coverage": len(selected) / len(test) if test else 0.0,
                "selective_risk": _mean(
                    float(row["net_gain"] <= 0.0) for row in selected
                ),
                "population_net_gain": sum(row["net_gain"] for row in selected)
                / len(test)
                if test
                else 0.0,
                "selected_mean_net_gain": _mean(row["net_gain"] for row in selected),
                "oracle_population_net_gain": sum(max(0.0, row["net_gain"]) for row in test)
                / len(test)
                if test
                else 0.0,
            },
            "test_rows": test,
        }
    return {
        "acceptance_partition": {
            "conformal_count": len(calibration_states),
            "certificate_count": len(certificate_states),
        },
        "methods": output,
    }


def _observation(state: RawGainState, method: str) -> ContinuousGainObservation:
    if method == "static":
        mean_gain = state.static_predicted_gain
        internal = external = interaction = 0.0
    elif method == "ensemble":
        mean_gain = state.ensemble_mean_gain - METHOD_COSTS[method]
        internal = state.ensemble_internal_variance
        external = interaction = 0.0
    else:
        mean_gain = state.crossed_mean_gain - METHOD_COSTS[method]
        internal = state.crossed_internal_variance
        external = state.crossed_external_variance
        interaction = state.crossed_interaction_variance
    return ContinuousGainObservation(
        record_id=state.record_id,
        benchmark="lfp_real_multitask_shared_calibration",
        mean_gain=mean_gain,
        internal_variance=internal,
        external_variance=external,
        interaction_variance=interaction,
        gain=_net_gain(state, method),
    )


def _net_gain(state: RawGainState, method: str) -> float:
    return state.actual_comparative_gain - METHOD_COSTS[method]


def _predict_rows(states, *, method, calibrator, conformal):
    observations = [_observation(state, method) for state in states]
    predictions = calibrator.predict(observations)
    output = []
    for state, observation, prediction in zip(states, observations, predictions):
        bound = conformal.predict_one(
            record_id=state.record_id,
            predicted_mean_gain=prediction.calibrated_mean_gain,
            predicted_scale=math.sqrt(prediction.calibrated_variance),
        )
        output.append(
            {
                "record_id": state.record_id,
                "benchmark": state.benchmark,
                "objective": state.objective,
                "raw_mean_gain": observation.mean_gain,
                "net_gain": observation.gain,
                "p_positive_gain": prediction.action_worthiness,
                "calibrated_mean_gain": prediction.calibrated_mean_gain,
                "calibrated_variance": prediction.calibrated_variance,
                "lower_gain_bound": bound.lower_gain_bound,
                "internal_variance": observation.internal_variance,
                "external_variance": observation.external_variance,
                "interaction_variance": observation.interaction_variance,
            }
        )
    return output


def _split_acceptance(states):
    calibration = []
    certificate = []
    for state in states:
        parity = int(hashlib.sha256(state.record_id.encode()).hexdigest()[-1], 16) % 2
        (calibration if parity == 0 else certificate).append(state)
    if not calibration or not certificate:
        raise ValueError("acceptance split must populate conformal and certificate subsets")
    return calibration, certificate


def _view_feature_names(records):
    component_names = sorted(
        {
            name
            for record in records
            for name in record.features
            if name.startswith("candidate_delta::component_fraction::")
        }
    )
    summary_names = sorted(
        {
            name
            for record in records
            for name in record.features
            if name
            in {
                "candidate_delta::composition_entropy",
                "candidate_delta::salt_fraction",
                "candidate_delta::additive_fraction",
            }
        }
    )
    solvent_salt = [
        name
        for name in component_names
        if name.rsplit("::", 1)[-1] in SOLVENTS_AND_SALTS
    ]
    additive = [
        name for name in component_names if name.rsplit("::", 1)[-1] in ADDITIVES
    ]
    return {
        "full": tuple(component_names + summary_names),
        "solvent_salt": tuple(
            solvent_salt
            + [name for name in summary_names if not name.endswith("additive_fraction")]
        ),
        "additive": tuple(
            additive
            + [name for name in summary_names if name.endswith("additive_fraction")]
        ),
    }


def _fit_linear(records, feature_names, alpha):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([_row(record, feature_names) for record in records], dtype=float)
    y = np.asarray([_target(record) for record in records], dtype=float)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return {"feature_names": feature_names, "scaler": scaler, "model": model}


def _predict_linear(model, record):
    import numpy as np

    x = np.asarray([_row(record, model["feature_names"])], dtype=float)
    return _bounded_gain(float(model["model"].predict(model["scaler"].transform(x))[0]))


def _fit_knn(records, feature_names, *, neighbors):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([_row(record, feature_names) for record in records], dtype=float)
    scaler = StandardScaler().fit(x)
    return {
        "feature_names": feature_names,
        "scaler": scaler,
        "x": scaler.transform(x),
        "y": tuple(_target(record) for record in records),
        "neighbors": min(neighbors, len(records)),
    }


def _predict_knn(model, record):
    import numpy as np

    x = np.asarray([_row(record, model["feature_names"])], dtype=float)
    query = model["scaler"].transform(x)[0]
    distances = np.sqrt(np.sum((model["x"] - query) ** 2, axis=1))
    order = np.argsort(distances)[: model["neighbors"]]
    selected_distances = distances[order]
    weights = 1.0 / np.maximum(selected_distances, 1e-6)
    targets = np.asarray([model["y"][int(index)] for index in order], dtype=float)
    return _bounded_gain(float(np.sum(weights * targets) / np.sum(weights)))


def _row(record, feature_names):
    return [float(record.features.get(name, 0.0)) for name in feature_names]


def _target(record):
    return float(record.metadata["hidden_outcome_for_evaluation_only"]["utility_difference"])


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _variance(values):
    rows = list(values)
    mean = _mean(rows)
    return _mean((value - mean) ** 2 for value in rows)


def _bounded_gain(value: float) -> float:
    return min(1.0, max(-1.0, value))
