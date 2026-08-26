from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .schema import ActionRecord


@dataclass(frozen=True)
class EvidenceResidualState:
    record_id: str
    benchmark: str
    objective: str
    candidate_group_ids: tuple[str, ...]
    model_success_probabilities: tuple[float, ...]
    proposal_aligned_source_matrix: tuple[tuple[float, ...], ...]
    source_names: tuple[str, ...]
    source_weights: tuple[float, ...]
    actual_net_gain: float

    def __post_init__(self) -> None:
        if len(self.model_success_probabilities) < 2:
            raise ValueError("at least two internal probability samples are required")
        if len(self.proposal_aligned_source_matrix) < 2:
            raise ValueError("at least two external bootstrap samples are required")
        if any(
            len(row) != len(self.source_names)
            for row in self.proposal_aligned_source_matrix
        ):
            raise ValueError("source matrix must align with source names")
        if len(self.source_weights) != len(self.source_names):
            raise ValueError("source weights must align with source names")


@dataclass(frozen=True)
class EvidenceResidualParameters:
    task_biases: dict[str, float]
    source_center: float
    source_scale: float
    source_coefficient: float
    blend_weight: float
    positive_intercept: float
    positive_slope: float
    negative_intercept: float
    negative_slope: float
    positive_residual_variance: float
    negative_residual_variance: float
    regularization: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceResidualPrediction:
    record_id: str
    benchmark: str
    direct_worthiness: float
    action_worthiness: float
    expected_net_gain: float
    positive_gain_mean: float
    negative_gain_mean: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    irreducible_variance: float
    total_variance: float
    weighted_source_margin: float
    source_support_probability: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class SafeEvidenceResidualModel:
    parameters: EvidenceResidualParameters

    def predict_one(self, state: EvidenceResidualState) -> EvidenceResidualPrediction:
        parameters = self.parameters
        external_values, external_weights = _external_samples(state)
        direct = tuple(_clip_probability(value) for value in state.model_success_probabilities)
        task_bias = parameters.task_biases.get(state.objective, 0.0)
        grid = []
        for probability in direct:
            direct_logit = _logit(probability)
            row = []
            for source_value in external_values:
                normalized = (
                    source_value - parameters.source_center
                ) / parameters.source_scale
                residual_probability = _sigmoid(
                    direct_logit
                    + task_bias
                    + parameters.source_coefficient * normalized
                )
                row.append(
                    (1.0 - parameters.blend_weight) * probability
                    + parameters.blend_weight * residual_probability
                )
            grid.append(tuple(row))
        decomposition = _weighted_grid_decomposition(grid, external_weights)
        weighted_margin = _weighted_mean(external_values, external_weights)
        magnitude = _weighted_mean(
            tuple(abs(value) for value in external_values), external_weights
        )
        positive_mean = max(
            1e-6,
            parameters.positive_intercept + parameters.positive_slope * magnitude,
        )
        negative_mean = max(
            1e-6,
            parameters.negative_intercept + parameters.negative_slope * magnitude,
        )
        worthiness = decomposition["mean"]
        expected_gain = worthiness * positive_mean - (1.0 - worthiness) * negative_mean
        gain_scale = positive_mean + negative_mean
        internal = gain_scale * gain_scale * decomposition["internal"]
        external = gain_scale * gain_scale * decomposition["external"]
        interaction = gain_scale * gain_scale * decomposition["interaction"]
        irreducible = (
            worthiness * parameters.positive_residual_variance
            + (1.0 - worthiness) * parameters.negative_residual_variance
            + worthiness * (positive_mean - expected_gain) ** 2
            + (1.0 - worthiness) * (-negative_mean - expected_gain) ** 2
        )
        total = irreducible + internal + external + interaction
        return EvidenceResidualPrediction(
            record_id=state.record_id,
            benchmark=state.benchmark,
            direct_worthiness=_mean(direct),
            action_worthiness=worthiness,
            expected_net_gain=expected_gain,
            positive_gain_mean=positive_mean,
            negative_gain_mean=negative_mean,
            internal_variance=internal,
            external_variance=external,
            interaction_variance=interaction,
            irreducible_variance=irreducible,
            total_variance=total,
            weighted_source_margin=weighted_margin,
            source_support_probability=_weighted_mean(
                tuple(float(value > 0.0) for value in external_values),
                external_weights,
            ),
        )

    def predict(
        self, states: Sequence[EvidenceResidualState]
    ) -> list[EvidenceResidualPrediction]:
        return [self.predict_one(state) for state in states]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "safe_crossed_evidence_residual_hurdle",
            "parameters": self.parameters.to_json(),
        }


def fit_safe_evidence_residual(
    states: Sequence[EvidenceResidualState],
    *,
    blend_weight: float,
    regularization: float,
) -> SafeEvidenceResidualModel:
    import numpy as np
    from scipy.optimize import minimize, nnls

    rows = list(states)
    if len(rows) < 8:
        raise ValueError("safe evidence residual fitting requires at least eight states")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend weight must lie in [0, 1]")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    objectives = tuple(sorted({state.objective for state in rows}))
    source_samples = [value for state in rows for value in _external_samples(state)[0]]
    source_center = _mean(source_samples)
    source_scale = max(1e-6, math.sqrt(_variance(source_samples)))

    def objective(theta) -> float:
        source_coefficient = _softplus(float(theta[0]))
        task_biases = dict(zip(objectives, (float(value) for value in theta[1:])))
        losses = []
        for state in rows:
            external_values, external_weights = _external_samples(state)
            probability = 0.0
            for direct_probability in state.model_success_probabilities:
                direct_probability = _clip_probability(direct_probability)
                direct_logit = _logit(direct_probability)
                for source_value, source_weight in zip(external_values, external_weights):
                    residual = _sigmoid(
                        direct_logit
                        + task_biases[state.objective]
                        + source_coefficient
                        * ((source_value - source_center) / source_scale)
                    )
                    probability += (
                        (1.0 - blend_weight) * direct_probability
                        + blend_weight * residual
                    ) * source_weight / len(state.model_success_probabilities)
            target = float(state.actual_net_gain > 0.0)
            probability = _clip_probability(probability)
            losses.append(
                -target * math.log(probability)
                - (1.0 - target) * math.log(1.0 - probability)
            )
        penalty = regularization * (
            source_coefficient * source_coefficient
            + sum(value * value for value in task_biases.values())
        ) / len(rows)
        return _mean(losses) + penalty

    starts = (
        np.zeros(1 + len(objectives)),
        np.asarray([-4.0] + [0.0] * len(objectives)),
        np.asarray([1.0] + [0.0] * len(objectives)),
    )
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=((-15.0, 10.0),) + ((-3.0, 3.0),) * len(objectives),
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not math.isfinite(float(best.fun)):
        raise RuntimeError("safe evidence residual optimization failed")
    source_coefficient = _softplus(float(best.x[0]))
    task_biases = dict(zip(objectives, (float(value) for value in best.x[1:])))

    magnitudes = np.asarray(
        [
            _weighted_mean(
                tuple(abs(value) for value in _external_samples(state)[0]),
                _external_samples(state)[1],
            )
            for state in rows
        ],
        dtype=float,
    )
    design = np.column_stack([np.ones(len(rows)), magnitudes])
    positive_indices = [index for index, state in enumerate(rows) if state.actual_net_gain > 0.0]
    negative_indices = [index for index, state in enumerate(rows) if state.actual_net_gain <= 0.0]
    if len(positive_indices) < 2 or len(negative_indices) < 2:
        raise ValueError("gain heads require at least two examples on each side of zero")
    positive_parameters, _ = nnls(
        design[positive_indices],
        np.asarray([rows[index].actual_net_gain for index in positive_indices]),
    )
    negative_parameters, _ = nnls(
        design[negative_indices],
        np.asarray([-rows[index].actual_net_gain for index in negative_indices]),
    )
    positive_prediction = design[positive_indices] @ positive_parameters
    negative_prediction = design[negative_indices] @ negative_parameters
    positive_targets = np.asarray(
        [rows[index].actual_net_gain for index in positive_indices]
    )
    negative_targets = np.asarray(
        [-rows[index].actual_net_gain for index in negative_indices]
    )
    parameters = EvidenceResidualParameters(
        task_biases=task_biases,
        source_center=source_center,
        source_scale=source_scale,
        source_coefficient=source_coefficient,
        blend_weight=blend_weight,
        positive_intercept=float(positive_parameters[0]),
        positive_slope=float(positive_parameters[1]),
        negative_intercept=float(negative_parameters[0]),
        negative_slope=float(negative_parameters[1]),
        positive_residual_variance=max(
            1e-8, float(np.mean((positive_targets - positive_prediction) ** 2))
        ),
        negative_residual_variance=max(
            1e-8, float(np.mean((negative_targets - negative_prediction) ** 2))
        ),
        regularization=regularization,
    )
    return SafeEvidenceResidualModel(parameters)


@dataclass(frozen=True)
class SafeAggregateEvidenceParameters:
    objectives: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    residual_bias: float
    feature_coefficients: tuple[float, ...]
    task_coefficients: dict[str, float]
    blend_weight: float
    positive_intercept: float
    positive_slope: float
    negative_intercept: float
    negative_slope: float
    positive_residual_variance: float
    negative_residual_variance: float
    regularization: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SafeAggregateEvidenceModel:
    parameters: SafeAggregateEvidenceParameters

    def probability_grid(
        self, state: EvidenceResidualState
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        direct = _mean(state.model_success_probabilities)
        features = _aggregate_features(state)
        residual = _aggregate_residual_probability(
            direct,
            features,
            state.objective,
            self.parameters,
        )
        worthiness = (
            (1.0 - self.parameters.blend_weight) * direct
            + self.parameters.blend_weight * residual
        )
        external_values, external_weights = _external_samples(state)
        raw_grid = []
        for probability in state.model_success_probabilities:
            row = []
            for source_value in external_values:
                pseudo_features = (
                    source_value,
                    source_value,
                    float(source_value > 0.0),
                    0.0,
                    0.0,
                    0.0,
                    abs(source_value),
                    *(source_value for _ in state.source_names),
                )
                pseudo_residual = _aggregate_residual_probability(
                    probability,
                    pseudo_features,
                    state.objective,
                    self.parameters,
                )
                row.append(
                    (1.0 - self.parameters.blend_weight) * probability
                    + self.parameters.blend_weight * pseudo_residual
                )
            raw_grid.append(tuple(row))
        return (
            _moment_match_probability_grid(raw_grid, external_weights, worthiness),
            external_weights,
        )

    def predict_one(self, state: EvidenceResidualState) -> EvidenceResidualPrediction:
        grid, external_weights = self.probability_grid(state)
        decomposition = _weighted_grid_decomposition(grid, external_weights)
        worthiness = decomposition["mean"]
        external_values, _ = _external_samples(state)
        magnitude = _weighted_mean(
            tuple(abs(value) for value in external_values), external_weights
        )
        positive_mean = max(
            1e-6,
            self.parameters.positive_intercept
            + self.parameters.positive_slope * magnitude,
        )
        negative_mean = max(
            1e-6,
            self.parameters.negative_intercept
            + self.parameters.negative_slope * magnitude,
        )
        expected_gain = worthiness * positive_mean - (1.0 - worthiness) * negative_mean
        gain_scale = positive_mean + negative_mean
        internal = gain_scale * gain_scale * decomposition["internal"]
        external = gain_scale * gain_scale * decomposition["external"]
        interaction = gain_scale * gain_scale * decomposition["interaction"]
        irreducible = (
            worthiness * self.parameters.positive_residual_variance
            + (1.0 - worthiness) * self.parameters.negative_residual_variance
            + worthiness * (positive_mean - expected_gain) ** 2
            + (1.0 - worthiness) * (-negative_mean - expected_gain) ** 2
        )
        return EvidenceResidualPrediction(
            record_id=state.record_id,
            benchmark=state.benchmark,
            direct_worthiness=_mean(state.model_success_probabilities),
            action_worthiness=worthiness,
            expected_net_gain=expected_gain,
            positive_gain_mean=positive_mean,
            negative_gain_mean=negative_mean,
            internal_variance=internal,
            external_variance=external,
            interaction_variance=interaction,
            irreducible_variance=irreducible,
            total_variance=irreducible + internal + external + interaction,
            weighted_source_margin=_weighted_mean(external_values, external_weights),
            source_support_probability=_weighted_mean(
                tuple(float(value > 0.0) for value in external_values),
                external_weights,
            ),
        )

    def predict(
        self, states: Sequence[EvidenceResidualState]
    ) -> list[EvidenceResidualPrediction]:
        return [self.predict_one(state) for state in states]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "safe_aggregate_evidence_residual_with_moment_matched_crossed_grid",
            "parameters": self.parameters.to_json(),
        }


@dataclass(frozen=True)
class TemperatureCalibratedEvidenceModel:
    base_model: SafeAggregateEvidenceModel
    temperature: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def predict_one(self, state: EvidenceResidualState) -> EvidenceResidualPrediction:
        raw_grid, external_weights = self.base_model.probability_grid(state)
        grid = tuple(
            tuple(_sigmoid(_logit(value) / self.temperature) for value in row)
            for row in raw_grid
        )
        decomposition = _weighted_grid_decomposition(grid, external_weights)
        worthiness = decomposition["mean"]
        external_values, _ = _external_samples(state)
        magnitude = _weighted_mean(
            tuple(abs(value) for value in external_values), external_weights
        )
        parameters = self.base_model.parameters
        positive_mean = max(
            1e-6,
            parameters.positive_intercept + parameters.positive_slope * magnitude,
        )
        negative_mean = max(
            1e-6,
            parameters.negative_intercept + parameters.negative_slope * magnitude,
        )
        expected_gain = worthiness * positive_mean - (1.0 - worthiness) * negative_mean
        gain_scale = positive_mean + negative_mean
        internal = gain_scale * gain_scale * decomposition["internal"]
        external = gain_scale * gain_scale * decomposition["external"]
        interaction = gain_scale * gain_scale * decomposition["interaction"]
        irreducible = (
            worthiness * parameters.positive_residual_variance
            + (1.0 - worthiness) * parameters.negative_residual_variance
            + worthiness * (positive_mean - expected_gain) ** 2
            + (1.0 - worthiness) * (-negative_mean - expected_gain) ** 2
        )
        return EvidenceResidualPrediction(
            record_id=state.record_id,
            benchmark=state.benchmark,
            direct_worthiness=_mean(state.model_success_probabilities),
            action_worthiness=worthiness,
            expected_net_gain=expected_gain,
            positive_gain_mean=positive_mean,
            negative_gain_mean=negative_mean,
            internal_variance=internal,
            external_variance=external,
            interaction_variance=interaction,
            irreducible_variance=irreducible,
            total_variance=irreducible + internal + external + interaction,
            weighted_source_margin=_weighted_mean(external_values, external_weights),
            source_support_probability=_weighted_mean(
                tuple(float(value > 0.0) for value in external_values),
                external_weights,
            ),
        )

    def predict(
        self, states: Sequence[EvidenceResidualState]
    ) -> list[EvidenceResidualPrediction]:
        return [self.predict_one(state) for state in states]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "temperature_calibrated_safe_aggregate_evidence_residual",
            "temperature": self.temperature,
            "base_model": self.base_model.to_json(),
        }


def fit_evidence_temperature(
    model: SafeAggregateEvidenceModel,
    states: Sequence[EvidenceResidualState],
    *,
    temperatures: Sequence[float] = tuple(step / 20.0 for step in range(2, 201)),
) -> TemperatureCalibratedEvidenceModel:
    rows = list(states)
    if not rows:
        raise ValueError("temperature calibration requires non-empty states")
    labels = [float(state.actual_net_gain > 0.0) for state in rows]
    raw_grids = [model.probability_grid(state) for state in rows]
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in temperatures:
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError("candidate temperatures must be finite and positive")
        probabilities = []
        for (grid, weights) in raw_grids:
            calibrated = tuple(
                tuple(_sigmoid(_logit(value) / temperature) for value in row)
                for row in grid
            )
            probabilities.append(_weighted_grid_decomposition(calibrated, weights)["mean"])
        loss = -_mean(
            label * math.log(_clip_probability(probability))
            + (1.0 - label) * math.log(_clip_probability(1.0 - probability))
            for label, probability in zip(labels, probabilities)
        )
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return TemperatureCalibratedEvidenceModel(model, best_temperature)


def fit_safe_aggregate_evidence_residual(
    states: Sequence[EvidenceResidualState],
    *,
    blend_weight: float,
    regularization: float,
) -> SafeAggregateEvidenceModel:
    import numpy as np
    from scipy.optimize import minimize, nnls

    rows = list(states)
    if len(rows) < 8:
        raise ValueError("safe aggregate evidence fitting requires at least eight states")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend weight must lie in [0, 1]")
    objectives = tuple(sorted({state.objective for state in rows}))
    matrix = np.asarray([_aggregate_features(state) for state in rows], dtype=float)
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    task_matrix = np.asarray(
        [
            [float(state.objective == objective) for objective in objectives[:-1]]
            for state in rows
        ],
        dtype=float,
    )
    direct = np.asarray(
        [_mean(state.model_success_probabilities) for state in rows], dtype=float
    )
    target = np.asarray([float(state.actual_net_gain > 0.0) for state in rows])

    def objective(theta) -> float:
        bias = float(theta[0])
        feature_coefficients = theta[1 : 1 + standardized.shape[1]]
        task_coefficients = theta[1 + standardized.shape[1] :]
        residual = np.asarray(
            [
                _sigmoid(
                    _logit(probability)
                    + bias
                    + float(features @ feature_coefficients)
                    + float(tasks @ task_coefficients)
                )
                for probability, features, tasks in zip(
                    direct, standardized, task_matrix
                )
            ]
        )
        probability = (1.0 - blend_weight) * direct + blend_weight * residual
        probability = np.clip(probability, 1e-9, 1.0 - 1e-9)
        loss = -np.mean(
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)
        )
        penalty = regularization * float(theta @ theta) / len(rows)
        return float(loss + penalty)

    parameter_count = 1 + standardized.shape[1] + task_matrix.shape[1]
    result = minimize(
        objective,
        np.zeros(parameter_count),
        method="L-BFGS-B",
        bounds=((-3.0, 3.0),) * parameter_count,
    )
    if not math.isfinite(float(result.fun)):
        raise RuntimeError("safe aggregate evidence optimization failed")
    theta = result.x
    feature_count = standardized.shape[1]
    feature_coefficients = tuple(float(value) for value in theta[1 : 1 + feature_count])
    task_coefficients = {
        objective: float(value)
        for objective, value in zip(objectives[:-1], theta[1 + feature_count :])
    }
    task_coefficients[objectives[-1]] = 0.0

    magnitudes = np.asarray(
        [
            _weighted_mean(
                tuple(abs(value) for value in _external_samples(state)[0]),
                _external_samples(state)[1],
            )
            for state in rows
        ],
        dtype=float,
    )
    design = np.column_stack([np.ones(len(rows)), magnitudes])
    positive_indices = [index for index, state in enumerate(rows) if state.actual_net_gain > 0.0]
    negative_indices = [index for index, state in enumerate(rows) if state.actual_net_gain <= 0.0]
    if len(positive_indices) < 2 or len(negative_indices) < 2:
        raise ValueError("gain heads require at least two examples on each side of zero")
    positive_parameters, _ = nnls(
        design[positive_indices],
        np.asarray([rows[index].actual_net_gain for index in positive_indices]),
    )
    negative_parameters, _ = nnls(
        design[negative_indices],
        np.asarray([-rows[index].actual_net_gain for index in negative_indices]),
    )
    positive_targets = np.asarray(
        [rows[index].actual_net_gain for index in positive_indices]
    )
    negative_targets = np.asarray(
        [-rows[index].actual_net_gain for index in negative_indices]
    )
    positive_prediction = design[positive_indices] @ positive_parameters
    negative_prediction = design[negative_indices] @ negative_parameters
    return SafeAggregateEvidenceModel(
        SafeAggregateEvidenceParameters(
            objectives=objectives,
            feature_mean=tuple(float(value) for value in feature_mean),
            feature_scale=tuple(float(value) for value in feature_scale),
            residual_bias=float(theta[0]),
            feature_coefficients=feature_coefficients,
            task_coefficients=task_coefficients,
            blend_weight=blend_weight,
            positive_intercept=float(positive_parameters[0]),
            positive_slope=float(positive_parameters[1]),
            negative_intercept=float(negative_parameters[0]),
            negative_slope=float(negative_parameters[1]),
            positive_residual_variance=max(
                1e-8, float(np.mean((positive_targets - positive_prediction) ** 2))
            ),
            negative_residual_variance=max(
                1e-8, float(np.mean((negative_targets - negative_prediction) ** 2))
            ),
            regularization=regularization,
        )
    )


def build_evidence_residual_states(
    records: Sequence[ActionRecord],
    raw_states: Sequence[dict[str, object]],
    llm_probabilities_a: dict[str, tuple[float, ...]],
    *,
    source_names: Sequence[str],
    source_weights: dict[str, float],
    method_cost: float = 0.01,
) -> list[EvidenceResidualState]:
    raw_by_id = {str(state["record_id"]): state for state in raw_states}
    names = tuple(source_names)
    weights = tuple(float(source_weights[name]) for name in names)
    output = []
    for record in records:
        probabilities_a = tuple(llm_probabilities_a[record.record_id])
        proposal_sign = 1.0 if _mean(probabilities_a) >= 0.5 else -1.0
        success_probabilities = tuple(
            probability if proposal_sign > 0.0 else 1.0 - probability
            for probability in probabilities_a
        )
        raw = raw_by_id[record.record_id]
        static_sign = float(raw["proposal_sign"])
        aligned_matrix = tuple(
            tuple(proposal_sign * static_sign * float(value) for value in row)
            for row in raw["source_matrix"]
        )
        utility_difference = float(
            record.metadata["hidden_outcome_for_evaluation_only"]["utility_difference"]
        )
        output.append(
            EvidenceResidualState(
                record_id=record.record_id,
                benchmark=record.benchmark,
                objective=str(record.metadata["objective"]),
                candidate_group_ids=tuple(
                    str(value) for value in record.metadata["candidate_group_ids"]
                ),
                model_success_probabilities=success_probabilities,
                proposal_aligned_source_matrix=aligned_matrix,
                source_names=names,
                source_weights=weights,
                actual_net_gain=proposal_sign * utility_difference - method_cost,
            )
        )
    return output


def aggregate_probability_grid_decomposition(
    model: SafeAggregateEvidenceModel,
    state: EvidenceResidualState,
) -> dict[str, float]:
    prediction = model.predict_one(state)
    external_values, external_weights = _external_samples(state)
    raw_grid = []
    for probability in state.model_success_probabilities:
        row = []
        for source_value in external_values:
            pseudo_features = (
                source_value,
                source_value,
                float(source_value > 0.0),
                0.0,
                0.0,
                0.0,
                abs(source_value),
                *(source_value for _ in state.source_names),
            )
            residual = _aggregate_residual_probability(
                probability,
                pseudo_features,
                state.objective,
                model.parameters,
            )
            row.append(
                (1.0 - model.parameters.blend_weight) * probability
                + model.parameters.blend_weight * residual
            )
        raw_grid.append(tuple(row))
    grid = _moment_match_probability_grid(
        raw_grid, external_weights, prediction.action_worthiness
    )
    return _weighted_grid_decomposition(grid, external_weights)


def _aggregate_features(state: EvidenceResidualState) -> tuple[float, ...]:
    matrix = state.proposal_aligned_source_matrix
    source_means = tuple(
        _mean(row[source_index] for row in matrix)
        for source_index in range(len(state.source_names))
    )
    normalized_source_weights = _normalized_weights(state.source_weights)
    external_values, external_weights = _external_samples(state)
    return (
        _mean(source_means),
        _weighted_mean(source_means, normalized_source_weights),
        _weighted_mean(
            tuple(float(value > 0.0) for value in external_values),
            external_weights,
        ),
        math.sqrt(_weighted_variance(source_means, normalized_source_weights)),
        math.sqrt(_weighted_variance(external_values, external_weights)),
        math.sqrt(_variance(state.model_success_probabilities)),
        abs(_mean(source_means)),
        *source_means,
    )


def _aggregate_residual_probability(
    direct_probability: float,
    features: Sequence[float],
    objective: str,
    parameters: SafeAggregateEvidenceParameters,
) -> float:
    standardized = tuple(
        (value - center) / scale
        for value, center, scale in zip(
            features, parameters.feature_mean, parameters.feature_scale
        )
    )
    return _sigmoid(
        _logit(direct_probability)
        + parameters.residual_bias
        + sum(
            value * coefficient
            for value, coefficient in zip(
                standardized, parameters.feature_coefficients
            )
        )
        + parameters.task_coefficients.get(objective, 0.0)
    )


def _moment_match_probability_grid(
    grid: Sequence[Sequence[float]],
    external_weights: Sequence[float],
    target_mean: float,
) -> tuple[tuple[float, ...], ...]:
    weights = _normalized_weights(external_weights)
    logits = tuple(
        tuple(_logit(value) for value in row)
        for row in grid
    )
    lower = -30.0
    upper = 30.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        mean = _mean(
            _weighted_mean(
                tuple(_sigmoid(value + midpoint) for value in row), weights
            )
            for row in logits
        )
        if mean < target_mean:
            lower = midpoint
        else:
            upper = midpoint
    shift = 0.5 * (lower + upper)
    matched = tuple(
        tuple(_sigmoid(value + shift) for value in row)
        for row in logits
    )
    observed = _mean(_weighted_mean(row, weights) for row in matched)
    if not math.isclose(observed, target_mean, abs_tol=1e-10):
        raise AssertionError("moment-matched grid does not preserve worthiness mean")
    return matched


def probability_grid_decomposition(
    model: SafeEvidenceResidualModel,
    state: EvidenceResidualState,
) -> dict[str, float]:
    parameters = model.parameters
    external_values, external_weights = _external_samples(state)
    grid = []
    for direct_probability in state.model_success_probabilities:
        direct_probability = _clip_probability(direct_probability)
        row = []
        for source_value in external_values:
            residual = _sigmoid(
                _logit(direct_probability)
                + parameters.task_biases.get(state.objective, 0.0)
                + parameters.source_coefficient
                * ((source_value - parameters.source_center) / parameters.source_scale)
            )
            row.append(
                (1.0 - parameters.blend_weight) * direct_probability
                + parameters.blend_weight * residual
            )
        grid.append(tuple(row))
    return _weighted_grid_decomposition(grid, external_weights)


def _external_samples(
    state: EvidenceResidualState,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    source_total = sum(state.source_weights)
    if source_total <= 0.0:
        raise ValueError("source weights must have positive mass")
    normalized_source_weights = tuple(value / source_total for value in state.source_weights)
    replicas = len(state.proposal_aligned_source_matrix)
    values = tuple(
        value
        for row in state.proposal_aligned_source_matrix
        for value in row
    )
    weights = tuple(
        normalized_source_weights[source_index] / replicas
        for _ in state.proposal_aligned_source_matrix
        for source_index in range(len(state.source_names))
    )
    return values, weights


def _weighted_grid_decomposition(
    grid: Sequence[Sequence[float]],
    external_weights: Sequence[float],
) -> dict[str, float]:
    rows = tuple(tuple(float(value) for value in row) for row in grid)
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("crossed grid requires at least two internal and external samples")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("crossed grid must be rectangular")
    if len(external_weights) != len(rows[0]):
        raise ValueError("external weights must align with grid columns")
    weights = _normalized_weights(external_weights)
    grand = _mean(_weighted_mean(row, weights) for row in rows)
    row_means = tuple(_weighted_mean(row, weights) for row in rows)
    column_means = tuple(_mean(row[index] for row in rows) for index in range(len(weights)))
    internal = _mean((value - grand) ** 2 for value in row_means)
    external = _weighted_mean(
        tuple((value - grand) ** 2 for value in column_means), weights
    )
    interaction = _mean(
        _weighted_mean(
            tuple(
                (
                    rows[row_index][column_index]
                    - row_means[row_index]
                    - column_means[column_index]
                    + grand
                )
                ** 2
                for column_index in range(len(weights))
            ),
            weights,
        )
        for row_index in range(len(rows))
    )
    total = _mean(
        _weighted_mean(tuple((value - grand) ** 2 for value in row), weights)
        for row in rows
    )
    if not math.isclose(total, internal + external + interaction, abs_tol=1e-12):
        raise AssertionError("weighted crossed decomposition is not exact")
    return {
        "mean": grand,
        "internal": internal,
        "external": external,
        "interaction": interaction,
        "total": total,
    }


def _normalized_weights(values: Sequence[float]) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if any(value < 0.0 or not math.isfinite(value) for value in weights):
        raise ValueError("weights must be finite and nonnegative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    return tuple(value / total for value in weights)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights))


def _weighted_variance(values: Sequence[float], weights: Sequence[float]) -> float:
    center = _weighted_mean(values, weights)
    return sum(weight * (value - center) ** 2 for value, weight in zip(values, weights))


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _variance(values) -> float:
    rows = list(values)
    center = _mean(rows)
    return _mean((value - center) ** 2 for value in rows)


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def _logit(value: float) -> float:
    probability = _clip_probability(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))
