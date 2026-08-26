from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from typing import Any

from .features import Standardizer, feature_names_from_records
from .schema import ActionRecord


@dataclass(frozen=True)
class GainConfig:
    cost_weight: float = 0.15
    failure_harm_weight: float = 0.25
    worthiness_margin: float = 0.0
    utility_mode: str = "success_gated"


@dataclass(frozen=True)
class GainTarget:
    record_id: str
    benchmark: str
    normalized_utility: float
    normalized_cost: float
    failure_harm: float
    base_gain: float
    reservation_value: float
    gain: float
    worthy: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GainPrediction:
    record_id: str
    mean_gain: float
    gain_scale: float
    action_worthiness: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmpiricalUtilityNormalizer:
    values_by_task: dict[str, list[float]]

    @classmethod
    def fit(cls, records: list[ActionRecord]) -> "EmpiricalUtilityNormalizer":
        values: dict[str, list[float]] = {}
        for record in records:
            values.setdefault(record.benchmark, []).append(float(record.utility))
        return cls({task: sorted(task_values) for task, task_values in values.items()})

    def encode(self, record: ActionRecord) -> float:
        return self.encode_value(record.benchmark, float(record.utility))

    def encode_value(self, benchmark: str, utility: float) -> float:
        values = self.values_by_task.get(benchmark)
        if not values:
            raise ValueError(f"no train utility distribution for {benchmark}")
        rank = bisect.bisect_right(values, float(utility))
        return (rank + 0.5) / (len(values) + 1.0)

    def to_json(self) -> dict[str, Any]:
        return {
            task: {
                "count": len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for task, values in self.values_by_task.items()
        }


@dataclass(frozen=True)
class BudgetReservationValues:
    budget_fraction: float
    values_by_task: dict[str, float]
    outside_option: float = 0.0
    strategy: str = "outcome_conditioned_base_gain"

    @classmethod
    def fit(
        cls,
        records: list[ActionRecord],
        normalizer: EmpiricalUtilityNormalizer,
        config: GainConfig,
        *,
        budget_fraction: float,
        outside_option: float = 0.0,
        strategy: str = "outcome_conditioned_base_gain",
    ) -> "BudgetReservationValues":
        if not 0.0 < budget_fraction < 1.0:
            raise ValueError("budget_fraction must lie in (0, 1)")
        if strategy not in {
            "outcome_conditioned_base_gain",
            "potential_success_slot_value",
        }:
            raise ValueError(f"unknown reservation strategy: {strategy}")
        gains: dict[str, list[float]] = {}
        for record in records:
            if strategy == "outcome_conditioned_base_gain":
                value = _base_gain(record, normalizer, config)[0]
            else:
                cost = _clamp(record.features.get("cost", 0.5))
                value = normalizer.encode(record) - config.cost_weight * cost
            gains.setdefault(record.benchmark, []).append(value)
        return cls(
            budget_fraction=budget_fraction,
            values_by_task={
                task: max(outside_option, _quantile(values, 1.0 - budget_fraction))
                for task, values in gains.items()
            },
            outside_option=outside_option,
            strategy=strategy,
        )

    def value(self, benchmark: str) -> float:
        if benchmark not in self.values_by_task:
            raise ValueError(f"no reservation value for {benchmark}")
        return self.values_by_task[benchmark]

    def to_json(self) -> dict[str, Any]:
        return {
            "budget_fraction": self.budget_fraction,
            "outside_option": self.outside_option,
            "strategy": self.strategy,
            "values_by_task": dict(self.values_by_task),
        }


@dataclass(frozen=True)
class EstimatedGainProjector:
    normalizer: EmpiricalUtilityNormalizer
    reservations: BudgetReservationValues
    config: GainConfig

    def project(
        self,
        record: ActionRecord,
        *,
        estimated_utility: float,
        success_probability: float,
    ) -> float:
        utility = self.normalizer.encode_value(record.benchmark, estimated_utility)
        cost = _clamp(record.features.get("cost", 0.5))
        irreversibility = 1.0 - _clamp(record.features.get("reversibility", 0.5))
        failure_harm = 0.5 * cost + 0.5 * irreversibility
        probability = _clamp(success_probability)
        return (
            probability * utility
            - self.config.cost_weight * cost
            - self.config.failure_harm_weight * (1.0 - probability) * failure_harm
            - self.reservations.value(record.benchmark)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "train_only_scientific_utility_to_gain_projector",
            "normalizer": self.normalizer.to_json(),
            "reservations": self.reservations.to_json(),
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class ScientificShortfallScales:
    lower_quantile: float
    scales_by_task: dict[str, float]
    lower_references_by_task: dict[str, float]

    @classmethod
    def fit(
        cls,
        train_targets: list[GainTarget],
        *,
        lower_quantile: float = 0.05,
    ) -> "ScientificShortfallScales":
        if not 0.0 <= lower_quantile < 1.0:
            raise ValueError("lower_quantile must lie in [0, 1)")
        tasks = sorted({target.benchmark for target in train_targets})
        scales: dict[str, float] = {}
        references: dict[str, float] = {}
        for task in tasks:
            task_targets = [target for target in train_targets if target.benchmark == task]
            base_gains = sorted(target.base_gain for target in task_targets)
            if not base_gains:
                raise ValueError(f"no shortfall targets for task {task}")
            index = min(len(base_gains) - 1, max(0, math.floor(lower_quantile * len(base_gains))))
            lower_reference = base_gains[index]
            reservation_values = {target.reservation_value for target in task_targets}
            if len(reservation_values) != 1:
                raise ValueError(f"task {task} has inconsistent reservation values")
            reservation = next(iter(reservation_values))
            references[task] = lower_reference
            scales[task] = max(1e-6, reservation - lower_reference)
        return cls(lower_quantile, scales, references)

    def loss(self, target: GainTarget) -> float:
        if target.benchmark not in self.scales_by_task:
            raise ValueError(f"no shortfall scale for {target.benchmark}")
        return min(1.0, max(0.0, -target.gain / self.scales_by_task[target.benchmark]))

    def to_json(self) -> dict[str, Any]:
        return {
            "lower_quantile": self.lower_quantile,
            "scales_by_task": dict(self.scales_by_task),
            "lower_references_by_task": dict(self.lower_references_by_task),
        }


def gain_targets(
    records: list[ActionRecord],
    normalizer: EmpiricalUtilityNormalizer,
    config: GainConfig,
    reservation_values: BudgetReservationValues | None = None,
) -> list[GainTarget]:
    targets: list[GainTarget] = []
    for record in records:
        base_gain, utility, cost, failure_harm = _base_gain(record, normalizer, config)
        reservation = reservation_values.value(record.benchmark) if reservation_values else 0.0
        gain = base_gain - reservation
        targets.append(
            GainTarget(
                record_id=record.record_id,
                benchmark=record.benchmark,
                normalized_utility=utility,
                normalized_cost=cost,
                failure_harm=failure_harm,
                base_gain=base_gain,
                reservation_value=reservation,
                gain=gain,
                worthy=int(gain > config.worthiness_margin),
            )
        )
    return targets


@dataclass
class GainDistributionModel:
    feature_names: list[str]
    standardizer: Standardizer
    weights: list[float]
    bias: float
    residual_scale: float
    temperature: float = 1.0

    def predict(self, records: list[ActionRecord]) -> list[GainPrediction]:
        rows = self.standardizer.transform([_row(record, self.feature_names) for record in records])
        predictions: list[GainPrediction] = []
        scale = max(1e-6, self.residual_scale * self.temperature)
        for record, row in zip(records, rows):
            mean_gain = self.bias + sum(weight * value for weight, value in zip(self.weights, row))
            probability = _normal_cdf(mean_gain / scale)
            predictions.append(
                GainPrediction(
                    record_id=record.record_id,
                    mean_gain=mean_gain,
                    gain_scale=scale,
                    action_worthiness=probability,
                )
            )
        return predictions

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "gain_distribution_model",
            "feature_names": list(self.feature_names),
            "standardizer": self.standardizer.to_json(),
            "weights": list(self.weights),
            "bias": self.bias,
            "residual_scale": self.residual_scale,
            "temperature": self.temperature,
        }


def fit_gain_distribution(
    train_records: list[ActionRecord],
    train_targets: list[GainTarget],
    calibration_records: list[ActionRecord],
    calibration_targets: list[GainTarget],
    *,
    feature_names: list[str] | None = None,
    l2: float = 0.1,
) -> GainDistributionModel:
    if len(train_records) != len(train_targets):
        raise ValueError("train records and gain targets must align")
    if len(calibration_records) != len(calibration_targets):
        raise ValueError("calibration records and gain targets must align")
    names = feature_names or feature_names_from_records(train_records)
    standardizer = Standardizer().fit([_row(record, names) for record in train_records])
    rows = standardizer.transform([_row(record, names) for record in train_records])
    weights, bias = _fit_ridge(rows, [target.gain for target in train_targets], l2=l2)
    residuals = [
        target.gain - (bias + sum(weight * value for weight, value in zip(weights, row)))
        for row, target in zip(rows, train_targets)
    ]
    residual_scale = math.sqrt(sum(value * value for value in residuals) / max(1, len(residuals)))
    model = GainDistributionModel(names, standardizer, weights, bias, max(residual_scale, 1e-4))
    model.temperature = _calibrate_temperature(model, calibration_records, calibration_targets)
    return model


def _calibrate_temperature(
    model: GainDistributionModel,
    records: list[ActionRecord],
    targets: list[GainTarget],
) -> float:
    if not records:
        return 1.0
    original = model.temperature
    best_temperature = 1.0
    best_loss = float("inf")
    for step in range(2, 201):
        temperature = step / 20.0
        model.temperature = temperature
        probabilities = [prediction.action_worthiness for prediction in model.predict(records)]
        loss = _log_loss([target.worthy for target in targets], probabilities)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    model.temperature = original
    return best_temperature


def _row(record: ActionRecord, feature_names: list[str]) -> list[float]:
    return [float(record.features.get(name, 0.0)) for name in feature_names]


def _fit_ridge(rows: list[list[float]], targets: list[float], *, l2: float) -> tuple[list[float], float]:
    if not rows:
        raise ValueError("cannot fit gain model on empty records")
    dimension = len(rows[0]) + 1
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    vector = [0.0 for _ in range(dimension)]
    for row, target in zip(rows, targets):
        values = [1.0] + list(row)
        for left in range(dimension):
            vector[left] += values[left] * target
            for right in range(dimension):
                matrix[left][right] += values[left] * values[right]
    for index in range(1, dimension):
        matrix[index][index] += l2
    solution = _solve(matrix, vector)
    return solution[1:], solution[0]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) < 1e-10:
            augmented[column][column] = 1e-10
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return [augmented[index][-1] for index in range(size)]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _log_loss(labels: list[int], probabilities: list[float]) -> float:
    total = 0.0
    for label, probability in zip(labels, probabilities):
        probability = min(1.0 - 1e-12, max(1e-12, probability))
        total += -(label * math.log(probability) + (1 - label) * math.log(1.0 - probability))
    return total / max(1, len(labels))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _base_gain(
    record: ActionRecord,
    normalizer: EmpiricalUtilityNormalizer,
    config: GainConfig,
) -> tuple[float, float, float, float]:
    utility = normalizer.encode(record)
    cost = _clamp(record.features.get("cost", 0.5))
    irreversibility = 1.0 - _clamp(record.features.get("reversibility", 0.5))
    failure_harm = 0.5 * cost + 0.5 * irreversibility
    if config.utility_mode == "success_gated":
        realized_utility = float(record.label) * utility
    elif config.utility_mode == "continuous_realized":
        realized_utility = utility
    else:
        raise ValueError(f"unknown utility mode: {config.utility_mode}")
    value = (
        realized_utility
        - config.cost_weight * cost
        - config.failure_harm_weight * float(1 - record.label) * failure_harm
    )
    return value, utility, cost, failure_harm


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile from empty values")
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
