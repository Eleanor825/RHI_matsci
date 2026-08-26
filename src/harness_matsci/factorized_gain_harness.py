from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class FactorizedGainState:
    record_id: str
    benchmark: str
    objective: str
    group_id: str
    model_worthiness_probabilities: tuple[float, ...]
    model_success_probabilities: tuple[float, ...]
    model_utility_means: tuple[float, ...]
    source_names: tuple[str, ...]
    source_success_probabilities: tuple[float, ...]
    source_utility_features: tuple[tuple[float, float, float], ...]
    source_weights: tuple[float, ...]
    normalized_cost: float
    failure_harm: float
    reservation_value: float
    actual_success: int
    actual_normalized_utility: float
    actual_net_gain: float

    def __post_init__(self) -> None:
        internal_count = len(self.model_worthiness_probabilities)
        if internal_count < 2:
            raise ValueError("factorized harness requires at least two internal replicas")
        if len(self.model_success_probabilities) != internal_count:
            raise ValueError("model success probabilities must align")
        if len(self.model_utility_means) != internal_count:
            raise ValueError("model utility means must align")
        external_count = len(self.source_names)
        if external_count < 1:
            raise ValueError("factorized harness requires at least one external source")
        if len(self.source_success_probabilities) != external_count:
            raise ValueError("source success probabilities must align")
        if len(self.source_utility_features) != external_count:
            raise ValueError("source utility features must align")
        if len(self.source_weights) != external_count:
            raise ValueError("source weights must align")
        if self.actual_success not in (0, 1):
            raise ValueError("actual success must be binary")


@dataclass(frozen=True)
class FactorizedGainParameters:
    objectives: tuple[str, ...]
    source_names: tuple[str, ...]
    source_utility_coefficients: dict[str, tuple[float, ...]]
    success_coefficients: tuple[float, ...]
    utility_coefficients: tuple[float, ...]
    utility_residual_scales: dict[str, float]
    blend_weight: float
    cost_weight: float
    failure_harm_weight: float
    regularization: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FactorizedGainPrediction:
    record_id: str
    benchmark: str
    action_worthiness: float
    expected_net_gain: float
    success_probability: float
    conditional_utility_mean: float
    probability_internal_variance: float
    probability_external_variance: float
    probability_interaction_variance: float
    gain_internal_variance: float
    gain_external_variance: float
    gain_interaction_variance: float
    aleatoric_gain_variance: float
    total_gain_variance: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class FactorizedGainHarness:
    parameters: FactorizedGainParameters
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def grids(
        self, state: FactorizedGainState
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[float, ...],
    ]:
        source_utilities = tuple(
            _source_utility(state, source_index, self.parameters)
            for source_index in range(len(state.source_names))
        )
        probability_grid = []
        expected_gain_grid = []
        success_grid = []
        utility_grid = []
        for model_index in range(len(state.model_success_probabilities)):
            probability_row = []
            gain_row = []
            success_row = []
            utility_row = []
            for source_index in range(len(state.source_names)):
                success_probability = _success_probability(
                    state,
                    model_index,
                    source_index,
                    self.parameters,
                )
                utility_mean = _conditional_utility(
                    state,
                    model_index,
                    source_index,
                    source_utilities[source_index],
                    self.parameters,
                )
                scale = self.parameters.utility_residual_scales[state.objective]
                success_threshold = (
                    state.reservation_value
                    + self.parameters.cost_weight * state.normalized_cost
                )
                conditional_tail = 1.0 - _normal_cdf(
                    (success_threshold - utility_mean) / max(1e-6, scale)
                )
                structured_probability = success_probability * conditional_tail
                direct_probability = state.model_worthiness_probabilities[model_index]
                worthiness = (
                    (1.0 - self.parameters.blend_weight) * direct_probability
                    + self.parameters.blend_weight * structured_probability
                )
                worthiness = _sigmoid(_logit(worthiness) / self.temperature)
                expected_gain = (
                    success_probability * utility_mean
                    - self.parameters.cost_weight * state.normalized_cost
                    - self.parameters.failure_harm_weight
                    * (1.0 - success_probability)
                    * state.failure_harm
                    - state.reservation_value
                )
                probability_row.append(worthiness)
                gain_row.append(expected_gain)
                success_row.append(success_probability)
                utility_row.append(utility_mean)
            probability_grid.append(tuple(probability_row))
            expected_gain_grid.append(tuple(gain_row))
            success_grid.append(tuple(success_row))
            utility_grid.append(tuple(utility_row))
        weights = _normalized_weights(state.source_weights)
        return (
            tuple(probability_grid),
            tuple(expected_gain_grid),
            tuple(success_grid),
            weights,
        )

    def predict_one(self, state: FactorizedGainState) -> FactorizedGainPrediction:
        probability_grid, gain_grid, success_grid, weights = self.grids(state)
        probability = crossed_grid_decomposition(probability_grid, weights)
        gain = crossed_grid_decomposition(gain_grid, weights)
        success = crossed_grid_decomposition(success_grid, weights)["mean"]
        source_utilities = tuple(
            _source_utility(state, index, self.parameters)
            for index in range(len(state.source_names))
        )
        utility_grid = tuple(
            tuple(
                _conditional_utility(
                    state,
                    model_index,
                    source_index,
                    source_utilities[source_index],
                    self.parameters,
                )
                for source_index in range(len(state.source_names))
            )
            for model_index in range(len(state.model_success_probabilities))
        )
        utility_mean = crossed_grid_decomposition(utility_grid, weights)["mean"]
        scale = self.parameters.utility_residual_scales[state.objective]
        aleatoric = (
            success * scale * scale
            + success * (1.0 - success) * (
                utility_mean + self.parameters.failure_harm_weight * state.failure_harm
            ) ** 2
        )
        total = aleatoric + gain["internal"] + gain["external"] + gain["interaction"]
        return FactorizedGainPrediction(
            record_id=state.record_id,
            benchmark=state.benchmark,
            action_worthiness=probability["mean"],
            expected_net_gain=gain["mean"],
            success_probability=success,
            conditional_utility_mean=utility_mean,
            probability_internal_variance=probability["internal"],
            probability_external_variance=probability["external"],
            probability_interaction_variance=probability["interaction"],
            gain_internal_variance=gain["internal"],
            gain_external_variance=gain["external"],
            gain_interaction_variance=gain["interaction"],
            aleatoric_gain_variance=aleatoric,
            total_gain_variance=total,
        )

    def predict(
        self, states: Sequence[FactorizedGainState]
    ) -> list[FactorizedGainPrediction]:
        return [self.predict_one(state) for state in states]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "factorized_success_conditional_utility_gain_harness",
            "temperature": self.temperature,
            "parameters": self.parameters.to_json(),
        }


def fit_factorized_gain_harness(
    states: Sequence[FactorizedGainState],
    *,
    blend_weight: float,
    regularization: float,
    cost_weight: float = 0.15,
    failure_harm_weight: float = 0.25,
) -> FactorizedGainHarness:
    import numpy as np
    from scipy.optimize import minimize

    rows = list(states)
    if len(rows) < 12:
        raise ValueError("factorized harness fitting requires at least twelve states")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend weight must lie in [0, 1]")
    objectives = tuple(sorted({state.objective for state in rows}))
    source_names = rows[0].source_names
    if any(state.source_names != source_names for state in rows):
        raise ValueError("all states must share a source contract")

    source_utility_coefficients = {}
    for objective in objectives:
        task_rows = [
            state
            for state in rows
            if state.objective == objective and state.actual_success == 1
        ]
        if len(task_rows) < 4:
            raise ValueError(
                f"conditional utility fitting requires four successes for {objective}"
            )
        for source_index, source_name in enumerate(source_names):
            design = np.asarray(
                [
                    (1.0, *state.source_utility_features[source_index])
                    for state in task_rows
                ],
                dtype=float,
            )
            target = np.asarray(
                [state.actual_normalized_utility for state in task_rows], dtype=float
            )
            penalty = regularization * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            coefficients = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ target,
            )
            source_utility_coefficients[f"{objective}::{source_name}"] = tuple(
                float(value) for value in coefficients
            )

    success_features = []
    success_targets = []
    utility_features = []
    utility_targets = []
    for state in rows:
        task_vector = tuple(float(state.objective == task) for task in objectives[:-1])
        source_utilities = tuple(
            _source_utility_with_coefficients(
                state,
                source_index,
                source_utility_coefficients,
            )
            for source_index in range(len(source_names))
        )
        for model_index in range(len(state.model_success_probabilities)):
            for source_index in range(len(source_names)):
                source_vector = tuple(
                    float(source_index == index)
                    for index in range(len(source_names) - 1)
                )
                success_features.append(
                    (
                        1.0,
                        _logit(state.model_success_probabilities[model_index]),
                        _logit(state.source_success_probabilities[source_index]),
                        *task_vector,
                        *source_vector,
                    )
                )
                success_targets.append(float(state.actual_success))
                if state.actual_success == 1:
                    utility_features.append(
                        (
                            1.0,
                            state.model_utility_means[model_index],
                            source_utilities[source_index],
                            *task_vector,
                            *source_vector,
                        )
                    )
                    utility_targets.append(state.actual_normalized_utility)

    success_matrix = np.asarray(success_features, dtype=float)
    success_target = np.asarray(success_targets, dtype=float)

    def success_objective(theta) -> float:
        logits = np.clip(success_matrix @ theta, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
        loss = -np.mean(
            success_target * np.log(probabilities)
            + (1.0 - success_target) * np.log(1.0 - probabilities)
        )
        penalty = regularization * float(theta[1:] @ theta[1:]) / len(rows)
        return float(loss + penalty)

    success_result = minimize(
        success_objective,
        np.zeros(success_matrix.shape[1]),
        method="L-BFGS-B",
        bounds=((-6.0, 6.0),) * success_matrix.shape[1],
    )
    if not math.isfinite(float(success_result.fun)):
        raise RuntimeError("success factor optimization failed")

    utility_matrix = np.asarray(utility_features, dtype=float)
    utility_target = np.asarray(utility_targets, dtype=float)
    utility_penalty = regularization * np.eye(utility_matrix.shape[1])
    utility_penalty[0, 0] = 0.0
    utility_coefficients = np.linalg.solve(
        utility_matrix.T @ utility_matrix + utility_penalty,
        utility_matrix.T @ utility_target,
    )

    provisional = FactorizedGainHarness(
        FactorizedGainParameters(
            objectives=objectives,
            source_names=source_names,
            source_utility_coefficients=source_utility_coefficients,
            success_coefficients=tuple(float(value) for value in success_result.x),
            utility_coefficients=tuple(float(value) for value in utility_coefficients),
            utility_residual_scales={objective: 0.1 for objective in objectives},
            blend_weight=blend_weight,
            cost_weight=cost_weight,
            failure_harm_weight=failure_harm_weight,
            regularization=regularization,
        )
    )
    residuals = {objective: [] for objective in objectives}
    for state in rows:
        if state.actual_success != 1:
            continue
        source_utilities = tuple(
            _source_utility(state, index, provisional.parameters)
            for index in range(len(source_names))
        )
        means = [
            _conditional_utility(
                state,
                model_index,
                source_index,
                source_utilities[source_index],
                provisional.parameters,
            )
            for model_index in range(len(state.model_utility_means))
            for source_index in range(len(source_names))
        ]
        residuals[state.objective].append(
            state.actual_normalized_utility - sum(means) / len(means)
        )
    scales = {
        objective: max(
            0.03,
            math.sqrt(sum(value * value for value in values) / len(values)),
        )
        for objective, values in residuals.items()
    }
    return FactorizedGainHarness(
        FactorizedGainParameters(
            **{
                **provisional.parameters.to_json(),
                "utility_residual_scales": scales,
            }
        )
    )


def fit_factorized_temperature(
    model: FactorizedGainHarness,
    states: Sequence[FactorizedGainState],
    *,
    temperatures: Sequence[float] = tuple(step / 20.0 for step in range(2, 201)),
) -> FactorizedGainHarness:
    rows = list(states)
    if not rows:
        raise ValueError("temperature calibration requires non-empty states")
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in temperatures:
        candidate = FactorizedGainHarness(model.parameters, float(temperature))
        probabilities = [row.action_worthiness for row in candidate.predict(rows)]
        labels = [float(state.actual_net_gain > 0.0) for state in rows]
        loss = -sum(
            label * math.log(_clip_probability(probability))
            + (1.0 - label) * math.log(_clip_probability(1.0 - probability))
            for label, probability in zip(labels, probabilities)
        ) / len(rows)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return FactorizedGainHarness(model.parameters, best_temperature)


def crossed_grid_decomposition(
    grid: Sequence[Sequence[float]],
    source_weights: Sequence[float],
) -> dict[str, float]:
    rows = tuple(tuple(float(value) for value in row) for row in grid)
    if len(rows) < 2 or not rows[0]:
        raise ValueError("crossed grid requires at least two internal rows and one source")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("crossed grid must be rectangular")
    weights = _normalized_weights(source_weights)
    if len(weights) != len(rows[0]):
        raise ValueError("source weights must align with grid columns")
    row_means = tuple(sum(value * weight for value, weight in zip(row, weights)) for row in rows)
    column_means = tuple(
        sum(row[column] for row in rows) / len(rows)
        for column in range(len(weights))
    )
    grand = sum(row_means) / len(row_means)
    internal = sum((value - grand) ** 2 for value in row_means) / len(row_means)
    external = sum(
        weight * (value - grand) ** 2
        for value, weight in zip(column_means, weights)
    )
    interaction = sum(
        sum(
            weight
            * (rows[row][column] - row_means[row] - column_means[column] + grand) ** 2
            for column, weight in enumerate(weights)
        )
        for row in range(len(rows))
    ) / len(rows)
    total = sum(
        sum(weight * (value - grand) ** 2 for value, weight in zip(row, weights))
        for row in rows
    ) / len(rows)
    if not math.isclose(total, internal + external + interaction, abs_tol=1e-12):
        raise AssertionError("crossed variance decomposition is not exact")
    return {
        "mean": grand,
        "internal": internal,
        "external": external,
        "interaction": interaction,
        "total": total,
    }


def _source_utility(
    state: FactorizedGainState,
    source_index: int,
    parameters: FactorizedGainParameters,
) -> float:
    return _source_utility_with_coefficients(
        state,
        source_index,
        parameters.source_utility_coefficients,
    )


def _source_utility_with_coefficients(state, source_index, coefficients_by_key):
    key = f"{state.objective}::{state.source_names[source_index]}"
    coefficients = coefficients_by_key[key]
    features = (1.0, *state.source_utility_features[source_index])
    return _clip_unit(sum(value * coefficient for value, coefficient in zip(features, coefficients)))


def _success_probability(state, model_index, source_index, parameters):
    task_vector = tuple(
        float(state.objective == task) for task in parameters.objectives[:-1]
    )
    source_vector = tuple(
        float(source_index == index)
        for index in range(len(parameters.source_names) - 1)
    )
    features = (
        1.0,
        _logit(state.model_success_probabilities[model_index]),
        _logit(state.source_success_probabilities[source_index]),
        *task_vector,
        *source_vector,
    )
    return _sigmoid(
        sum(value * coefficient for value, coefficient in zip(features, parameters.success_coefficients))
    )


def _conditional_utility(state, model_index, source_index, source_utility, parameters):
    task_vector = tuple(
        float(state.objective == task) for task in parameters.objectives[:-1]
    )
    source_vector = tuple(
        float(source_index == index)
        for index in range(len(parameters.source_names) - 1)
    )
    features = (
        1.0,
        state.model_utility_means[model_index],
        source_utility,
        *task_vector,
        *source_vector,
    )
    return _clip_unit(
        sum(value * coefficient for value, coefficient in zip(features, parameters.utility_coefficients))
    )


def _normalized_weights(values):
    weights = tuple(float(value) for value in values)
    if any(value < 0.0 or not math.isfinite(value) for value in weights):
        raise ValueError("source weights must be finite and nonnegative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("source weights must have positive mass")
    return tuple(value / total for value in weights)


def _clip_unit(value):
    return min(1.0, max(0.0, float(value)))


def _clip_probability(value):
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _logit(value):
    probability = _clip_probability(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
