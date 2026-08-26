from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReplicaWorthinessObservation:
    record_id: str
    benchmark: str
    mean_gain: float
    internal_variance: float
    external_variance: float
    worthy: int


@dataclass(frozen=True)
class SourceCalibrationParameters:
    bias: float
    base_variance: float
    internal_weight: float
    external_weight: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SourceDecomposedWorthinessCalibrator:
    parameters_by_task: dict[str, SourceCalibrationParameters]

    def predict_one(self, observation: ReplicaWorthinessObservation) -> float:
        if observation.benchmark not in self.parameters_by_task:
            raise ValueError(f"no source calibrator for task {observation.benchmark}")
        parameters = self.parameters_by_task[observation.benchmark]
        variance = (
            parameters.base_variance
            + parameters.internal_weight * max(0.0, observation.internal_variance)
            + parameters.external_weight * max(0.0, observation.external_variance)
        )
        z_score = (observation.mean_gain + parameters.bias) / math.sqrt(max(1e-12, variance))
        return min(1.0 - 1e-9, max(1e-9, _normal_cdf(z_score)))

    def predict(self, observations: Iterable[ReplicaWorthinessObservation]) -> list[float]:
        return [self.predict_one(observation) for observation in observations]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "source_decomposed_heteroscedastic_gain_calibrator",
            "parameters_by_task": {
                task: parameters.to_json()
                for task, parameters in sorted(self.parameters_by_task.items())
            },
        }


def fit_source_decomposed_calibrator(
    observations: list[ReplicaWorthinessObservation],
    *,
    bias_grid: tuple[float, ...] = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0),
    base_standard_deviation_grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2),
    source_weight_grid: tuple[float, ...] = (0.0, 0.25, 1.0, 4.0, 16.0),
    regularization: float = 1e-4,
) -> SourceDecomposedWorthinessCalibrator:
    if not observations:
        raise ValueError("source calibration requires observations")
    tasks = sorted({observation.benchmark for observation in observations})
    parameters = {}
    for task in tasks:
        task_observations = [observation for observation in observations if observation.benchmark == task]
        if len({observation.worthy for observation in task_observations}) < 2:
            raise ValueError(f"source calibration task {task} requires both worthy and unworthy examples")
        best_loss = float("inf")
        best_parameters = None
        for bias in bias_grid:
            for standard_deviation in base_standard_deviation_grid:
                for internal_weight in source_weight_grid:
                    for external_weight in source_weight_grid:
                        candidate = SourceCalibrationParameters(
                            bias=bias,
                            base_variance=standard_deviation**2,
                            internal_weight=internal_weight,
                            external_weight=external_weight,
                        )
                        loss = _task_loss(task_observations, candidate)
                        loss += regularization * (
                            bias * bias + internal_weight * internal_weight + external_weight * external_weight
                        )
                        if loss < best_loss:
                            best_loss = loss
                            best_parameters = candidate
        if best_parameters is None:
            raise RuntimeError(f"failed to fit source calibrator for task {task}")
        parameters[task] = best_parameters
    return SourceDecomposedWorthinessCalibrator(parameters)


def _task_loss(
    observations: list[ReplicaWorthinessObservation],
    parameters: SourceCalibrationParameters,
) -> float:
    loss = 0.0
    for observation in observations:
        variance = (
            parameters.base_variance
            + parameters.internal_weight * max(0.0, observation.internal_variance)
            + parameters.external_weight * max(0.0, observation.external_variance)
        )
        probability = _normal_cdf(
            (observation.mean_gain + parameters.bias) / math.sqrt(max(1e-12, variance))
        )
        probability = min(1.0 - 1e-9, max(1e-9, probability))
        loss -= observation.worthy * math.log(probability)
        loss -= (1 - observation.worthy) * math.log1p(-probability)
    return loss / len(observations)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
