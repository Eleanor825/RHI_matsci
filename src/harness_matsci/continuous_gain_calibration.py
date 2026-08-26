from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ContinuousGainObservation:
    record_id: str
    benchmark: str
    mean_gain: float
    internal_variance: float
    external_variance: float
    gain: float
    interaction_variance: float = 0.0


@dataclass(frozen=True)
class ContinuousGainParameters:
    mean_slope: float
    mean_bias: float
    base_variance: float
    internal_weight: float
    external_weight: float
    interaction_weight: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibratedGainPrediction:
    record_id: str
    benchmark: str
    calibrated_mean_gain: float
    irreducible_variance: float
    calibrated_internal_variance: float
    calibrated_external_variance: float
    calibrated_interaction_variance: float
    calibrated_variance: float
    action_worthiness: float
    expected_shortfall: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuousSourceGainCalibrator:
    parameters_by_task: dict[str, ContinuousGainParameters]
    uses_source_variance: bool

    def predict_one(self, observation: ContinuousGainObservation) -> CalibratedGainPrediction:
        parameters = self.parameters_by_task[observation.benchmark]
        calibrated_mean = parameters.mean_slope * observation.mean_gain + parameters.mean_bias
        internal_variance = 0.0
        external_variance = 0.0
        interaction_variance = 0.0
        if self.uses_source_variance:
            internal_variance = parameters.internal_weight * max(
                0.0, observation.internal_variance
            )
            external_variance = parameters.external_weight * max(
                0.0, observation.external_variance
            )
            interaction_variance = parameters.interaction_weight * max(
                0.0, observation.interaction_variance
            )
        variance = (
            parameters.base_variance
            + internal_variance
            + external_variance
            + interaction_variance
        )
        variance = max(1e-10, variance)
        scale = math.sqrt(variance)
        standardized_mean = calibrated_mean / scale
        worthiness = _normal_cdf(standardized_mean)
        expected_shortfall = scale * _normal_pdf(standardized_mean) - calibrated_mean * _normal_cdf(
            -standardized_mean
        )
        return CalibratedGainPrediction(
            record_id=observation.record_id,
            benchmark=observation.benchmark,
            calibrated_mean_gain=calibrated_mean,
            irreducible_variance=parameters.base_variance,
            calibrated_internal_variance=internal_variance,
            calibrated_external_variance=external_variance,
            calibrated_interaction_variance=interaction_variance,
            calibrated_variance=variance,
            action_worthiness=min(1.0 - 1e-9, max(1e-9, worthiness)),
            expected_shortfall=max(0.0, expected_shortfall),
        )

    def predict(self, observations: Iterable[ContinuousGainObservation]) -> list[CalibratedGainPrediction]:
        return [self.predict_one(observation) for observation in observations]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "continuous_source_decomposed_gaussian_gain_calibrator",
            "uses_source_variance": self.uses_source_variance,
            "parameters_by_task": {
                task: parameters.to_json()
                for task, parameters in sorted(self.parameters_by_task.items())
            },
        }


def fit_continuous_gain_calibrator(
    observations: list[ContinuousGainObservation],
    *,
    use_source_variance: bool = True,
    regularization: float = 1e-3,
) -> ContinuousSourceGainCalibrator:
    import numpy as np
    from scipy.optimize import minimize

    if not observations:
        raise ValueError("continuous gain calibration requires observations")
    parameters_by_task: dict[str, ContinuousGainParameters] = {}
    for task in sorted({observation.benchmark for observation in observations}):
        rows = [observation for observation in observations if observation.benchmark == task]
        mean_gain = np.asarray([row.mean_gain for row in rows], dtype=float)
        internal = np.asarray([max(0.0, row.internal_variance) for row in rows], dtype=float)
        external = np.asarray([max(0.0, row.external_variance) for row in rows], dtype=float)
        interaction = np.asarray(
            [max(0.0, row.interaction_variance) for row in rows], dtype=float
        )
        target = np.asarray([row.gain for row in rows], dtype=float)
        residual_scale = max(1e-4, float(np.std(target - mean_gain)))

        def objective(theta):
            slope = math.exp(float(theta[0]))
            bias = float(theta[1])
            base_variance = math.exp(float(theta[2]))
            internal_weight = _softplus(float(theta[3])) if use_source_variance else 0.0
            external_weight = _softplus(float(theta[4])) if use_source_variance else 0.0
            interaction_weight = _softplus(float(theta[5])) if use_source_variance else 0.0
            variance = (
                base_variance
                + internal_weight * internal
                + external_weight * external
                + interaction_weight * interaction
            )
            prediction = slope * mean_gain + bias
            residual = target - prediction
            nll = 0.5 * np.mean(np.log(variance) + residual * residual / variance)
            penalty = regularization * (
                float(theta[0]) ** 2
                + bias * bias
                + internal_weight * internal_weight
                + external_weight * external_weight
                + interaction_weight * interaction_weight
            )
            return float(nll + penalty)

        starts = (
            np.asarray([0.0, 0.0, math.log(residual_scale**2), -4.0, -4.0, -4.0]),
            np.asarray(
                [
                    math.log(0.5),
                    float(np.mean(target)),
                    math.log(residual_scale**2),
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            np.asarray(
                [
                    math.log(1.5),
                    0.0,
                    math.log(max(1e-4, float(np.var(target)))),
                    2.0,
                    2.0,
                    2.0,
                ]
            ),
        )
        best = None
        bounds = (
            (-5.0, 5.0),
            (-2.0, 2.0),
            (-15.0, 3.0),
            (-15.0, 10.0),
            (-15.0, 10.0),
            (-15.0, 10.0),
        )
        for start in starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000},
            )
            if best is None or result.fun < best.fun:
                best = result
        if best is None or not math.isfinite(float(best.fun)):
            raise RuntimeError(f"continuous calibration failed for task {task}")
        theta = best.x
        parameters_by_task[task] = ContinuousGainParameters(
            mean_slope=math.exp(float(theta[0])),
            mean_bias=float(theta[1]),
            base_variance=math.exp(float(theta[2])),
            internal_weight=_softplus(float(theta[3])) if use_source_variance else 0.0,
            external_weight=_softplus(float(theta[4])) if use_source_variance else 0.0,
            interaction_weight=_softplus(float(theta[5])) if use_source_variance else 0.0,
        )
    return ContinuousSourceGainCalibrator(parameters_by_task, use_source_variance)


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
