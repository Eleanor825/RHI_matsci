from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class HurdleGainObservation:
    record_id: str
    benchmark: str
    mean_gain: float
    internal_variance: float
    external_variance: float
    gain: float
    known_failure_gain: float | None = None
    interaction_variance: float = 0.0
    internal_boundary_instability: float = 0.0
    external_boundary_instability: float = 0.0


@dataclass(frozen=True)
class HeteroscedasticLocationParameters:
    mean_slope: float
    mean_bias: float
    base_variance: float
    internal_weight: float
    external_weight: float
    interaction_weight: float
    internal_boundary_weight: float
    external_boundary_weight: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HurdleGainParameters:
    positive_probability: HeteroscedasticLocationParameters
    positive_gain: HeteroscedasticLocationParameters
    negative_gain: HeteroscedasticLocationParameters

    def to_json(self) -> dict[str, dict[str, float]]:
        return {
            "positive_probability": self.positive_probability.to_json(),
            "positive_gain": self.positive_gain.to_json(),
            "negative_gain": self.negative_gain.to_json(),
        }


@dataclass(frozen=True)
class HurdleGainPrediction:
    record_id: str
    benchmark: str
    mean_gain: float
    p_positive_gain: float
    expected_shortfall: float
    tail_risk_score: float
    irreducible_variance: float
    calibrated_internal_variance: float
    calibrated_external_variance: float
    calibrated_variance: float
    positive_component_mean: float
    negative_component_mean: float
    calibrated_interaction_variance: float = 0.0
    calibrated_internal_boundary_instability: float = 0.0
    calibrated_external_boundary_instability: float = 0.0

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class HurdleSourceGainCalibrator:
    parameters_by_task: dict[str, HurdleGainParameters]
    uses_source_variance: bool
    uses_decision_instability: bool = False
    beta: float = 0.10
    tail_weight: float = 0.50
    gain_lower_bound: float = -1.40
    gain_upper_bound: float = 1.00

    def predict_one(self, observation: HurdleGainObservation) -> HurdleGainPrediction:
        parameters = self.parameters_by_task[observation.benchmark]
        positive_probability = _probit_probability(
            parameters.positive_probability,
            observation,
            use_source_variance=self.uses_source_variance,
            use_decision_instability=self.uses_decision_instability,
        )
        positive_mean, positive_parts = _component_prediction(
            parameters.positive_gain,
            observation,
            use_source_variance=self.uses_source_variance,
            use_decision_instability=self.uses_decision_instability,
        )
        negative_mean, negative_parts = _component_prediction(
            parameters.negative_gain,
            observation,
            use_source_variance=self.uses_source_variance,
            use_decision_instability=self.uses_decision_instability,
        )
        positive_mean = min(self.gain_upper_bound, max(1e-6, positive_mean))
        if observation.known_failure_gain is not None:
            negative_mean = float(observation.known_failure_gain)
            negative_parts = (1e-10, 0.0, 0.0, 0.0, 0.0, 0.0)
        negative_mean = max(self.gain_lower_bound, min(-1e-6, negative_mean))
        negative_probability = 1.0 - positive_probability
        mean_gain = positive_probability * positive_mean + negative_probability * negative_mean
        positive_variance = sum(positive_parts)
        negative_variance = sum(negative_parts)
        calibrated_variance = (
            positive_probability
            * (positive_variance + (positive_mean - mean_gain) ** 2)
            + negative_probability
            * (negative_variance + (negative_mean - mean_gain) ** 2)
        )
        irreducible_variance = (
            positive_probability * positive_parts[0]
            + negative_probability * negative_parts[0]
            + positive_probability * (positive_mean - mean_gain) ** 2
            + negative_probability * (negative_mean - mean_gain) ** 2
        )
        internal_variance = (
            positive_probability * positive_parts[1]
            + negative_probability * negative_parts[1]
        )
        external_variance = (
            positive_probability * positive_parts[2]
            + negative_probability * negative_parts[2]
        )
        interaction_variance = (
            positive_probability * positive_parts[3]
            + negative_probability * negative_parts[3]
        )
        internal_boundary_instability = (
            positive_probability * positive_parts[4]
            + negative_probability * negative_parts[4]
        )
        external_boundary_instability = (
            positive_probability * positive_parts[5]
            + negative_probability * negative_parts[5]
        )
        expected_shortfall = _mixture_expected_shortfall(
            positive_probability,
            positive_mean,
            math.sqrt(positive_variance),
            negative_mean,
            math.sqrt(negative_variance),
            beta=self.beta,
        )
        return HurdleGainPrediction(
            record_id=observation.record_id,
            benchmark=observation.benchmark,
            mean_gain=mean_gain,
            p_positive_gain=positive_probability,
            expected_shortfall=expected_shortfall,
            tail_risk_score=mean_gain - self.tail_weight * expected_shortfall,
            irreducible_variance=irreducible_variance,
            calibrated_internal_variance=(
                internal_variance + internal_boundary_instability
            ),
            calibrated_external_variance=(
                external_variance + external_boundary_instability
            ),
            calibrated_variance=max(1e-12, calibrated_variance),
            positive_component_mean=positive_mean,
            negative_component_mean=negative_mean,
            calibrated_interaction_variance=interaction_variance,
            calibrated_internal_boundary_instability=internal_boundary_instability,
            calibrated_external_boundary_instability=external_boundary_instability,
        )

    def predict(
        self, observations: Iterable[HurdleGainObservation]
    ) -> list[HurdleGainPrediction]:
        return [self.predict_one(observation) for observation in observations]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "source_decomposed_hurdle_gain_calibrator",
            "uses_source_variance": self.uses_source_variance,
            "uses_decision_instability": self.uses_decision_instability,
            "beta": self.beta,
            "tail_weight": self.tail_weight,
            "gain_bounds": [self.gain_lower_bound, self.gain_upper_bound],
            "parameters_by_task": {
                task: parameters.to_json()
                for task, parameters in sorted(self.parameters_by_task.items())
            },
        }


def fit_hurdle_gain_calibrator(
    observations: list[HurdleGainObservation],
    *,
    use_source_variance: bool = True,
    use_decision_instability: bool = False,
    beta: float = 0.10,
    tail_weight: float = 0.50,
    gain_lower_bound: float = -1.40,
    gain_upper_bound: float = 1.00,
    regularization: float = 1e-3,
) -> HurdleSourceGainCalibrator:
    if not observations:
        raise ValueError("hurdle gain calibration requires observations")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly between zero and one")
    if tail_weight < 0.0:
        raise ValueError("tail_weight must be nonnegative")
    if gain_lower_bound >= 0.0 or gain_upper_bound <= 0.0:
        raise ValueError("gain bounds must straddle zero")
    parameters_by_task = {}
    for task in sorted({observation.benchmark for observation in observations}):
        rows = [observation for observation in observations if observation.benchmark == task]
        positive_rows = [row for row in rows if row.gain > 0.0]
        negative_rows = [row for row in rows if row.gain <= 0.0]
        if len(positive_rows) < 2 or len(negative_rows) < 2:
            raise ValueError(f"task {task} requires at least two gains on each side of zero")
        parameters_by_task[task] = HurdleGainParameters(
            positive_probability=_fit_probit_parameters(
                rows,
                use_source_variance=use_source_variance,
                use_decision_instability=use_decision_instability,
                regularization=regularization,
            ),
            positive_gain=_fit_location_parameters(
                positive_rows,
                use_source_variance=use_source_variance,
                use_decision_instability=use_decision_instability,
                regularization=regularization,
            ),
            negative_gain=_fit_location_parameters(
                negative_rows,
                use_source_variance=use_source_variance,
                use_decision_instability=use_decision_instability,
                regularization=regularization,
            ),
        )
    return HurdleSourceGainCalibrator(
        parameters_by_task=parameters_by_task,
        uses_source_variance=use_source_variance,
        uses_decision_instability=use_decision_instability,
        beta=beta,
        tail_weight=tail_weight,
        gain_lower_bound=gain_lower_bound,
        gain_upper_bound=gain_upper_bound,
    )


def _fit_probit_parameters(
    rows: list[HurdleGainObservation],
    *,
    use_source_variance: bool,
    use_decision_instability: bool,
    regularization: float,
) -> HeteroscedasticLocationParameters:
    import numpy as np
    from scipy.optimize import minimize

    mean_gain = np.asarray([row.mean_gain for row in rows], dtype=float)
    internal = np.asarray([max(0.0, row.internal_variance) for row in rows], dtype=float)
    external = np.asarray([max(0.0, row.external_variance) for row in rows], dtype=float)
    interaction = np.asarray(
        [max(0.0, row.interaction_variance) for row in rows], dtype=float
    )
    internal_boundary = np.asarray(
        [max(0.0, row.internal_boundary_instability) for row in rows], dtype=float
    )
    external_boundary = np.asarray(
        [max(0.0, row.external_boundary_instability) for row in rows], dtype=float
    )
    labels = np.asarray([float(row.gain > 0.0) for row in rows], dtype=float)

    def objective(theta):
        slope = math.exp(float(theta[0]))
        bias = float(theta[1])
        base_variance = math.exp(float(theta[2]))
        internal_weight = _softplus(float(theta[3])) if use_source_variance else 0.0
        external_weight = _softplus(float(theta[4])) if use_source_variance else 0.0
        interaction_weight = _softplus(float(theta[5])) if use_source_variance else 0.0
        internal_boundary_weight = (
            _softplus(float(theta[6])) if use_decision_instability else 0.0
        )
        external_boundary_weight = (
            _softplus(float(theta[7])) if use_decision_instability else 0.0
        )
        variance = (
            base_variance
            + internal_weight * internal
            + external_weight * external
            + interaction_weight * interaction
            + internal_boundary_weight * internal_boundary
            + external_boundary_weight * external_boundary
        )
        z_score = (slope * mean_gain + bias) / np.sqrt(np.maximum(1e-12, variance))
        probabilities = np.asarray([_normal_cdf(float(value)) for value in z_score])
        probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
        log_loss = -np.mean(
            labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
        )
        penalty = regularization * (
            float(theta[0]) ** 2
            + bias * bias
            + internal_weight * internal_weight
            + external_weight * external_weight
            + interaction_weight * interaction_weight
            + internal_boundary_weight * internal_boundary_weight
            + external_boundary_weight * external_boundary_weight
        )
        return float(log_loss + penalty)

    prevalence = min(1.0 - 1e-4, max(1e-4, float(np.mean(labels))))
    starts = (
        np.asarray([0.0, _normal_ppf(prevalence), 0.0, -4.0, -4.0, -4.0, -4.0, -4.0]),
        np.asarray([math.log(0.5), _normal_ppf(prevalence), -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([math.log(1.5), 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    )
    return _optimize_parameters(objective, starts, label="probit")


def _fit_location_parameters(
    rows: list[HurdleGainObservation],
    *,
    use_source_variance: bool,
    use_decision_instability: bool,
    regularization: float,
) -> HeteroscedasticLocationParameters:
    import numpy as np

    mean_gain = np.asarray([row.mean_gain for row in rows], dtype=float)
    internal = np.asarray([max(0.0, row.internal_variance) for row in rows], dtype=float)
    external = np.asarray([max(0.0, row.external_variance) for row in rows], dtype=float)
    interaction = np.asarray(
        [max(0.0, row.interaction_variance) for row in rows], dtype=float
    )
    internal_boundary = np.asarray(
        [max(0.0, row.internal_boundary_instability) for row in rows], dtype=float
    )
    external_boundary = np.asarray(
        [max(0.0, row.external_boundary_instability) for row in rows], dtype=float
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
        internal_boundary_weight = (
            _softplus(float(theta[6])) if use_decision_instability else 0.0
        )
        external_boundary_weight = (
            _softplus(float(theta[7])) if use_decision_instability else 0.0
        )
        variance = (
            base_variance
            + internal_weight * internal
            + external_weight * external
            + interaction_weight * interaction
            + internal_boundary_weight * internal_boundary
            + external_boundary_weight * external_boundary
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
            + internal_boundary_weight * internal_boundary_weight
            + external_boundary_weight * external_boundary_weight
        )
        return float(nll + penalty)

    starts = (
        np.asarray([0.0, float(np.mean(target - mean_gain)), math.log(residual_scale**2), -4.0, -4.0, -4.0, -4.0, -4.0]),
        np.asarray([math.log(0.5), float(np.mean(target)), math.log(residual_scale**2), 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([math.log(1.5), 0.0, math.log(max(1e-4, float(np.var(target)))), 2.0, 2.0, 2.0, 2.0, 2.0]),
    )
    return _optimize_parameters(objective, starts, label="location")


def _optimize_parameters(objective, starts, *, label: str) -> HeteroscedasticLocationParameters:
    from scipy.optimize import minimize

    bounds = (
        (-5.0, 5.0),
        (-2.0, 2.0),
        (-15.0, 3.0),
        (-15.0, 10.0),
        (-15.0, 10.0),
        (-15.0, 10.0),
        (-15.0, 10.0),
        (-15.0, 10.0),
    )
    best = None
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
        raise RuntimeError(f"hurdle {label} calibration failed")
    theta = best.x
    return HeteroscedasticLocationParameters(
        mean_slope=math.exp(float(theta[0])),
        mean_bias=float(theta[1]),
        base_variance=math.exp(float(theta[2])),
        internal_weight=_softplus(float(theta[3])),
        external_weight=_softplus(float(theta[4])),
        interaction_weight=_softplus(float(theta[5])),
        internal_boundary_weight=_softplus(float(theta[6])),
        external_boundary_weight=_softplus(float(theta[7])),
    )


def _probit_probability(
    parameters: HeteroscedasticLocationParameters,
    observation: HurdleGainObservation,
    *,
    use_source_variance: bool,
    use_decision_instability: bool,
) -> float:
    variance = parameters.base_variance
    if use_source_variance:
        variance += parameters.internal_weight * max(0.0, observation.internal_variance)
        variance += parameters.external_weight * max(0.0, observation.external_variance)
        variance += parameters.interaction_weight * max(
            0.0, observation.interaction_variance
        )
    if use_decision_instability:
        variance += parameters.internal_boundary_weight * max(
            0.0, observation.internal_boundary_instability
        )
        variance += parameters.external_boundary_weight * max(
            0.0, observation.external_boundary_instability
        )
    location = parameters.mean_slope * observation.mean_gain + parameters.mean_bias
    probability = _normal_cdf(location / math.sqrt(max(1e-12, variance)))
    return min(1.0 - 1e-9, max(1e-9, probability))


def _component_prediction(
    parameters: HeteroscedasticLocationParameters,
    observation: HurdleGainObservation,
    *,
    use_source_variance: bool,
    use_decision_instability: bool,
) -> tuple[float, tuple[float, float, float, float, float, float]]:
    internal = 0.0
    external = 0.0
    interaction = 0.0
    internal_boundary = 0.0
    external_boundary = 0.0
    if use_source_variance:
        internal = parameters.internal_weight * max(0.0, observation.internal_variance)
        external = parameters.external_weight * max(0.0, observation.external_variance)
        interaction = parameters.interaction_weight * max(
            0.0, observation.interaction_variance
        )
    if use_decision_instability:
        internal_boundary = parameters.internal_boundary_weight * max(
            0.0, observation.internal_boundary_instability
        )
        external_boundary = parameters.external_boundary_weight * max(
            0.0, observation.external_boundary_instability
        )
    mean_gain = parameters.mean_slope * observation.mean_gain + parameters.mean_bias
    return mean_gain, (
        parameters.base_variance,
        internal,
        external,
        interaction,
        internal_boundary,
        external_boundary,
    )


def _mixture_expected_shortfall(
    positive_probability: float,
    positive_mean: float,
    positive_scale: float,
    negative_mean: float,
    negative_scale: float,
    *,
    beta: float,
) -> float:
    lower = min(
        negative_mean - 12.0 * negative_scale,
        positive_mean - 12.0 * positive_scale,
    )
    upper = max(
        negative_mean + 12.0 * negative_scale,
        positive_mean + 12.0 * positive_scale,
    )
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        probability = _mixture_cdf(
            midpoint,
            positive_probability,
            positive_mean,
            positive_scale,
            negative_mean,
            negative_scale,
        )
        if probability < beta:
            lower = midpoint
        else:
            upper = midpoint
    quantile = 0.5 * (lower + upper)
    positive_moment = _lower_normal_moment(quantile, positive_mean, positive_scale)
    negative_moment = _lower_normal_moment(quantile, negative_mean, negative_scale)
    lower_moment = (
        positive_probability * positive_moment
        + (1.0 - positive_probability) * negative_moment
    )
    return max(0.0, -lower_moment / beta)


def _mixture_cdf(
    value: float,
    positive_probability: float,
    positive_mean: float,
    positive_scale: float,
    negative_mean: float,
    negative_scale: float,
) -> float:
    return positive_probability * _normal_cdf((value - positive_mean) / positive_scale) + (
        1.0 - positive_probability
    ) * _normal_cdf((value - negative_mean) / negative_scale)


def _lower_normal_moment(value: float, mean: float, scale: float) -> float:
    z_score = (value - mean) / scale
    return mean * _normal_cdf(z_score) - scale * _normal_pdf(z_score)


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


def _normal_ppf(probability: float) -> float:
    from scipy.special import ndtri

    return float(ndtri(probability))
