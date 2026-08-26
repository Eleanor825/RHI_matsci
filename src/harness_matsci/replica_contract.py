from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Any

from .schema import ActionRecord
from .source_decomposition import SourceVariance, decompose_source_variance


EXTERNAL_VIEW_FEATURES = frozenset(
    {
        "evidence_support",
        "evidence_conflict",
        "source_reliability",
        "tool_agreement",
        "ood_score",
        "source_risk",
        "evidence_count",
    }
)


@dataclass(frozen=True)
class EvidenceView:
    view_id: str
    record: ActionRecord
    intervention: str
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "record_id": self.record.record_id,
            "intervention": self.intervention,
            "provenance": dict(self.provenance),
            "visible_context": self.record.visible_context,
            "evidence": list(self.record.evidence),
            "external_features": {
                key: self.record.features[key]
                for key in sorted(EXTERNAL_VIEW_FEATURES)
                if key in self.record.features
            },
        }


@dataclass(frozen=True)
class AgentEvidencePrediction:
    record_id: str
    agent_replica_id: str
    evidence_view_id: str
    predicted_gain: float
    positive_gain_probability: float
    backend: str
    internal_signals: dict[str, float]
    external_signals: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplicaGridSummary:
    record_id: str
    mean_predicted_gain: float
    mean_positive_gain_probability: float
    positive_gain_vote_fraction: float
    source_variance: SourceVariance

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "mean_predicted_gain": self.mean_predicted_gain,
            "mean_positive_gain_probability": self.mean_positive_gain_probability,
            "positive_gain_vote_fraction": self.positive_gain_vote_fraction,
            "source_variance": self.source_variance.to_json(),
        }


@dataclass(frozen=True)
class ActionReplicaGrid:
    record_id: str
    agent_replica_ids: tuple[str, ...]
    evidence_view_ids: tuple[str, ...]
    predictions: tuple[AgentEvidencePrediction, ...]

    @classmethod
    def from_predictions(
        cls, predictions: list[AgentEvidencePrediction]
    ) -> "ActionReplicaGrid":
        if not predictions:
            raise ValueError("replica grid requires predictions")
        record_ids = {prediction.record_id for prediction in predictions}
        if len(record_ids) != 1:
            raise ValueError("replica grid cannot mix action records")
        agents = tuple(sorted({prediction.agent_replica_id for prediction in predictions}))
        views = tuple(sorted({prediction.evidence_view_id for prediction in predictions}))
        if len(agents) < 2 or len(views) < 2:
            raise ValueError("replica grid requires at least two agents and two evidence views")
        cells = [(prediction.agent_replica_id, prediction.evidence_view_id) for prediction in predictions]
        if len(cells) != len(set(cells)):
            raise ValueError("replica grid contains duplicate agent/evidence cells")
        expected = {(agent, view) for agent in agents for view in views}
        if set(cells) != expected:
            missing = sorted(expected - set(cells))
            extra = sorted(set(cells) - expected)
            raise ValueError(f"replica grid is unbalanced: missing={missing}, extra={extra}")
        ordered = tuple(
            next(
                prediction
                for prediction in predictions
                if prediction.agent_replica_id == agent and prediction.evidence_view_id == view
            )
            for agent in agents
            for view in views
        )
        return cls(next(iter(record_ids)), agents, views, ordered)

    def predicted_gain_matrix(self) -> list[list[float]]:
        lookup = {
            (prediction.agent_replica_id, prediction.evidence_view_id): prediction.predicted_gain
            for prediction in self.predictions
        }
        return [
            [lookup[(agent, view)] for view in self.evidence_view_ids]
            for agent in self.agent_replica_ids
        ]

    def summarize(self) -> ReplicaGridSummary:
        gains = [prediction.predicted_gain for prediction in self.predictions]
        probabilities = [prediction.positive_gain_probability for prediction in self.predictions]
        return ReplicaGridSummary(
            record_id=self.record_id,
            mean_predicted_gain=mean(gains),
            mean_positive_gain_probability=mean(probabilities),
            positive_gain_vote_fraction=mean(float(value > 0.0) for value in gains),
            source_variance=decompose_source_variance(self.predicted_gain_matrix()),
        )


def controlled_evidence_views(record: ActionRecord) -> list[EvidenceView]:
    """Create label-blind evidence interventions for pipeline validation.

    These controlled views are not substitutes for real retrieval/tool replicas.
    They are deterministic interventions used to test whether source-decomposition
    code reacts to evidence removal and conflict without changing internal signals.
    """
    full = EvidenceView(
        view_id="evidence::full",
        record=record,
        intervention="none",
        provenance={"type": "observed_record", "label_blind": True},
    )
    reduced_evidence = list(record.evidence[:-1]) if len(record.evidence) > 1 else list(record.evidence)
    dropout_features = dict(record.features)
    if "evidence_support" in dropout_features:
        dropout_features["evidence_support"] *= 0.5
    if "evidence_count" in dropout_features:
        dropout_features["evidence_count"] *= 0.5
    dropout_features["ood_score"] = min(1.0, dropout_features.get("ood_score", 0.0) + 0.15)
    dropout = EvidenceView(
        view_id="evidence::dropout",
        record=replace(record, evidence=reduced_evidence, features=dropout_features),
        intervention="leave_one_evidence_out",
        provenance={"type": "controlled_ablation", "label_blind": True},
    )
    conflict_features = dict(record.features)
    conflict_features["evidence_conflict"] = max(0.75, conflict_features.get("evidence_conflict", 0.0))
    conflict_features["evidence_support"] = min(0.35, conflict_features.get("evidence_support", 0.35))
    conflict_features["tool_agreement"] = min(0.25, conflict_features.get("tool_agreement", 0.25))
    conflict_features["source_reliability"] = min(
        0.55, conflict_features.get("source_reliability", 0.55)
    )
    conflict = EvidenceView(
        view_id="evidence::conflict",
        record=replace(
            record,
            evidence=[*record.evidence, "Controlled challenge: an independent source disputes the estimate."],
            features=conflict_features,
        ),
        intervention="conflicting_source_injection",
        provenance={"type": "controlled_challenge", "label_blind": True},
    )
    return [full, dropout, conflict]
