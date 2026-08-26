from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PortfolioKnowledgeGradientRow:
    record_id: str
    benchmark: str
    transition: str
    features: Mapping[str, float]
    realized_incremental_gain: float


@dataclass(frozen=True)
class PortfolioKnowledgeGradientPrediction:
    record_id: str
    benchmark: str
    transition: str
    expected_incremental_gain: float
    internal_variance: float
    residual_variance: float
    predictive_variance: float
    model_replicas: int

    def to_json(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class _Replica:
    regressor: object


@dataclass(frozen=True)
class PortfolioKnowledgeGradientModel:
    feature_names_by_head: Mapping[str, tuple[str, ...]]
    replicas_by_head: Mapping[str, tuple[_Replica, ...]]
    residual_variance_by_head: Mapping[str, float]

    def predict(
        self,
        *,
        record_id: str,
        benchmark: str,
        transition: str,
        features: Mapping[str, float],
    ) -> PortfolioKnowledgeGradientPrediction:
        head = _head(benchmark, transition)
        names = self.feature_names_by_head[head]
        matrix = [_vectorize(features, names)]
        values = [
            float(replica.regressor.predict(matrix)[0])
            for replica in self.replicas_by_head[head]
        ]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        residual_variance = self.residual_variance_by_head[head]
        return PortfolioKnowledgeGradientPrediction(
            record_id=record_id,
            benchmark=benchmark,
            transition=transition,
            expected_incremental_gain=mean,
            internal_variance=variance,
            residual_variance=residual_variance,
            predictive_variance=variance + residual_variance,
            model_replicas=len(values),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "type": "task_transition_bootstrap_portfolio_knowledge_gradient",
            "heads": {
                head: {
                    "features": list(self.feature_names_by_head[head]),
                    "replicas": len(self.replicas_by_head[head]),
                }
                for head in sorted(self.feature_names_by_head)
            },
        }


def fit_portfolio_knowledge_gradient_model(
    rows: Iterable[PortfolioKnowledgeGradientRow],
    *,
    seed: int,
    replicas: int = 5,
    trees: int = 120,
    min_samples_leaf: int = 5,
) -> PortfolioKnowledgeGradientModel:
    observations = list(rows)
    if not observations:
        raise ValueError("portfolio knowledge-gradient fitting requires rows")
    if replicas < 2:
        raise ValueError("portfolio knowledge-gradient fitting needs replicas")
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    feature_names_by_head = {}
    replicas_by_head = {}
    residual_variance_by_head = {}
    heads = sorted({_head(row.benchmark, row.transition) for row in observations})
    for head_index, head in enumerate(heads):
        head_rows = [
            row
            for row in observations
            if _head(row.benchmark, row.transition) == head
        ]
        if len(head_rows) < 20:
            raise ValueError(f"portfolio knowledge-gradient head {head} needs 20 rows")
        feature_names = tuple(
            sorted({name for row in head_rows for name in row.features})
        )
        matrix = np.asarray(
            [_vectorize(row.features, feature_names) for row in head_rows],
            dtype=float,
        )
        targets = np.asarray(
            [float(row.realized_incremental_gain) for row in head_rows],
            dtype=float,
        )
        head_replicas = []
        for replica_index in range(replicas):
            replica_seed = seed * 1009 + head_index * 503 + replica_index * 97
            generator = np.random.default_rng(replica_seed)
            sampled = generator.choice(len(head_rows), size=len(head_rows), replace=True)
            regressor = ExtraTreesRegressor(
                n_estimators=trees,
                min_samples_leaf=min_samples_leaf,
                max_features=0.8,
                random_state=replica_seed,
                n_jobs=1,
            )
            regressor.fit(matrix[sampled], targets[sampled])
            head_replicas.append(_Replica(regressor))
        feature_names_by_head[head] = feature_names
        replicas_by_head[head] = tuple(head_replicas)
        fitted = np.mean(
            np.asarray(
                [replica.regressor.predict(matrix) for replica in head_replicas],
                dtype=float,
            ),
            axis=0,
        )
        residual_variance_by_head[head] = max(
            1e-10,
            float(np.mean((targets - fitted) ** 2)),
        )
    return PortfolioKnowledgeGradientModel(
        feature_names_by_head=feature_names_by_head,
        replicas_by_head=replicas_by_head,
        residual_variance_by_head=residual_variance_by_head,
    )


def portfolio_kg_features(
    *,
    current_score: float,
    shadow_price: float,
    structural_mean: float,
    structural_variance: float,
    internal_variance: float,
    external_variance: float,
    interaction_variance: float,
    positive_probability: float,
    evidence_cost: float,
    future_score_cells: Sequence[Sequence[float]],
) -> dict[str, float]:
    flattened = [float(value) for row in future_score_cells for value in row]
    if not flattened:
        raise ValueError("portfolio KG features require future score cells")
    crossing_probability = sum(value > shadow_price for value in flattened) / len(
        flattened
    )
    return {
        "current_score": float(current_score),
        "shadow_price": float(shadow_price),
        "distance_to_shadow": float(current_score) - float(shadow_price),
        "absolute_distance_to_shadow": abs(float(current_score) - float(shadow_price)),
        "structural_mean": float(structural_mean),
        "structural_standard_deviation": math.sqrt(max(0.0, structural_variance)),
        "internal_standard_deviation": math.sqrt(max(0.0, internal_variance)),
        "external_standard_deviation": math.sqrt(max(0.0, external_variance)),
        "interaction_standard_deviation": math.sqrt(max(0.0, interaction_variance)),
        "positive_probability": float(positive_probability),
        "evidence_cost": max(0.0, float(evidence_cost)),
        "future_crossing_probability": crossing_probability,
        "future_score_mean": sum(flattened) / len(flattened),
        "future_score_range": max(flattened) - min(flattened),
    }


def realized_single_acquisition_portfolio_gain(
    *,
    current_scores: Sequence[float],
    current_action_gains: Sequence[float],
    candidate_index: int,
    future_score: float,
    future_action_gain: float,
    budget: int,
    charged_evidence_cost: float,
) -> float:
    if len(current_scores) != len(current_action_gains):
        raise ValueError("scores and realized action gains must align")
    if not 0 <= candidate_index < len(current_scores):
        raise IndexError("candidate index is out of range")
    before = _selected_indices(current_scores, budget)
    future_scores = [float(value) for value in current_scores]
    future_scores[candidate_index] = float(future_score)
    after = _selected_indices(future_scores, budget)
    before_value = sum(float(current_action_gains[index]) for index in before)
    after_value = 0.0
    for index in after:
        after_value += (
            float(future_action_gain)
            if index == candidate_index
            else float(current_action_gains[index])
        )
    return after_value - before_value - max(0.0, float(charged_evidence_cost))


def deterministic_kg_partition(
    record_id: str,
    benchmark: str,
    *,
    seed: int,
) -> str:
    digest = hashlib.sha256(
        f"{seed}|portfolio-kg|{benchmark}|{record_id}".encode()
    ).hexdigest()
    return "fit" if int(digest[:16], 16) % 2 == 0 else "calibrate"


def _selected_indices(scores: Sequence[float], budget: int) -> set[int]:
    if budget < 1:
        raise ValueError("selection budget must be positive")
    eligible = [index for index, value in enumerate(scores) if float(value) > 0.0]
    eligible.sort(key=lambda index: (-float(scores[index]), index))
    return set(eligible[: min(budget, len(eligible))])


def _head(benchmark: str, transition: str) -> str:
    return f"{benchmark}::{transition}"


def _vectorize(features: Mapping[str, float], names: Sequence[str]) -> list[float]:
    output = []
    for name in names:
        value = float(features.get(name, 0.0))
        output.append(value if math.isfinite(value) else 0.0)
    return output
