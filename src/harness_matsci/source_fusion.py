from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .decision_instability import decision_boundary_instability


@dataclass(frozen=True)
class SourceFusionObservation:
    record_id: str
    benchmark: str
    source_names: tuple[str, ...]
    predicted_gain: tuple[tuple[float, ...], ...]
    gain: float

    def __post_init__(self) -> None:
        if len(self.predicted_gain) < 2:
            raise ValueError("at least two internal replicas are required")
        if any(len(row) != len(self.source_names) for row in self.predicted_gain):
            raise ValueError("source matrices must align with source names")


@dataclass(frozen=True)
class SourceFusionParameters:
    source_weights: dict[str, float]
    source_slopes: dict[str, float]
    source_biases: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FusedSourcePrediction:
    record_id: str
    benchmark: str
    mean_gain: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    internal_boundary_instability: float
    external_boundary_instability: float
    total_variance: float
    source_means: dict[str, float]
    source_weights: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceReliabilityFuser:
    parameters_by_task: dict[str, SourceFusionParameters]

    def predict_one(self, observation: SourceFusionObservation) -> FusedSourcePrediction:
        parameters = self.parameters_by_task[observation.benchmark]
        weights = [parameters.source_weights[source] for source in observation.source_names]
        slopes = [parameters.source_slopes[source] for source in observation.source_names]
        biases = [parameters.source_biases[source] for source in observation.source_names]
        source_means = [
            slopes[source_index]
            * (
                sum(row[source_index] for row in observation.predicted_gain)
                / len(observation.predicted_gain)
            )
            + biases[source_index]
            for source_index in range(len(observation.source_names))
        ]
        fused_replicas = [
            sum(
                weight * (slope * value + bias)
                for weight, slope, bias, value in zip(weights, slopes, biases, row)
            )
            for row in observation.predicted_gain
        ]
        mean_gain = sum(fused_replicas) / len(fused_replicas)
        internal_variance = _population_variance(fused_replicas)
        external_variance = sum(
            weight * (source_mean - mean_gain) ** 2
            for weight, source_mean in zip(weights, source_means)
        )
        calibrated_matrix = [
            [
                slope * value + bias
                for slope, bias, value in zip(slopes, biases, row)
            ]
            for row in observation.predicted_gain
        ]
        interaction_variance = sum(
            weight
            * (
                calibrated_matrix[replica_index][source_index]
                - fused_replicas[replica_index]
                - source_means[source_index]
                + mean_gain
            )
            ** 2
            for replica_index in range(len(calibrated_matrix))
            for source_index, weight in enumerate(weights)
        ) / len(calibrated_matrix)
        internal_boundary = decision_boundary_instability(fused_replicas)
        external_boundary = decision_boundary_instability(
            source_means,
            weights=weights,
        )
        return FusedSourcePrediction(
            record_id=observation.record_id,
            benchmark=observation.benchmark,
            mean_gain=mean_gain,
            internal_variance=internal_variance,
            external_variance=external_variance,
            interaction_variance=interaction_variance,
            internal_boundary_instability=internal_boundary.boundary_instability,
            external_boundary_instability=external_boundary.boundary_instability,
            total_variance=(
                internal_variance + external_variance + interaction_variance
            ),
            source_means=dict(zip(observation.source_names, source_means)),
            source_weights=dict(parameters.source_weights),
        )

    def predict(self, observations: list[SourceFusionObservation]) -> list[FusedSourcePrediction]:
        return [self.predict_one(observation) for observation in observations]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "task_conditional_reliability_weighted_source_fusion",
            "parameters_by_task": {
                task: parameters.to_json()
                for task, parameters in sorted(self.parameters_by_task.items())
            },
        }


def fit_source_reliability_fuser(
    observations: list[SourceFusionObservation],
    *,
    regularization: float = 1e-3,
    fit_affine: bool = True,
) -> SourceReliabilityFuser:
    if not observations:
        raise ValueError("source fusion requires observations")
    import numpy as np
    from scipy.optimize import minimize

    parameters_by_task = {}
    for task in sorted({observation.benchmark for observation in observations}):
        rows = [observation for observation in observations if observation.benchmark == task]
        source_names = rows[0].source_names
        if any(row.source_names != source_names for row in rows):
            raise ValueError(f"task {task} has inconsistent source ordering")
        matrix = np.asarray(
            [
                [
                    sum(replica[source_index] for replica in row.predicted_gain)
                    / len(row.predicted_gain)
                    for source_index in range(len(source_names))
                ]
                for row in rows
            ],
            dtype=float,
        )
        targets = np.asarray([row.gain for row in rows], dtype=float)

        source_count = len(source_names)

        def objective(theta):
            weights = _softmax(theta[:source_count])
            if fit_affine:
                slopes = np.exp(theta[source_count : 2 * source_count])
                biases = theta[2 * source_count : 3 * source_count]
            else:
                slopes = np.ones(source_count)
                biases = np.zeros(source_count)
            calibrated = matrix * slopes + biases
            residual = targets - (calibrated @ weights)
            penalty = regularization * float(
                np.sum((weights - 1.0 / len(weights)) ** 2)
            )
            if fit_affine:
                penalty += regularization * (
                    float(np.sum(theta[source_count : 2 * source_count] ** 2))
                    + float(np.sum(biases * biases))
                )
            return float(np.mean(residual * residual) + penalty)

        parameter_count = 3 * source_count if fit_affine else source_count
        starts = [np.zeros(parameter_count)]
        starts.extend(
            np.asarray(
                [
                    3.0 if index == source_index else -1.0
                    for index in range(len(source_names))
                ]
                + ([0.0] * 2 * source_count if fit_affine else [])
            )
            for source_index in range(len(source_names))
        )
        bounds = [(-10.0, 10.0)] * source_count
        if fit_affine:
            bounds += [(-8.0, 8.0)] * source_count + [(-2.0, 2.0)] * source_count
        best = None
        for start in starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if best is None or result.fun < best.fun:
                best = result
        if best is None or not math.isfinite(float(best.fun)):
            raise RuntimeError(f"source fusion failed for task {task}")
        weights = _softmax(best.x[:source_count])
        if fit_affine:
            slopes = [
                math.exp(float(value))
                for value in best.x[source_count : 2 * source_count]
            ]
            biases = [
                float(value) for value in best.x[2 * source_count : 3 * source_count]
            ]
        else:
            slopes = [1.0] * source_count
            biases = [0.0] * source_count
        parameters_by_task[task] = SourceFusionParameters(
            source_weights={
                source: float(weight) for source, weight in zip(source_names, weights)
            },
            source_slopes=dict(zip(source_names, slopes)),
            source_biases=dict(zip(source_names, biases)),
        )
    return SourceReliabilityFuser(parameters_by_task)


def _softmax(values):
    import numpy as np

    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    exponential = np.exp(shifted)
    return exponential / float(np.sum(exponential))


def _population_variance(values: list[float]) -> float:
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / len(values)
