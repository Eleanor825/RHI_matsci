from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any

from .features import Standardizer, feature_names_from_records
from .replica_contract import EvidenceView
from .schema import ActionRecord


EXCLUDED_RETRIEVAL_FEATURES = frozenset(
    {
        "verbal_confidence",
        "perturbation_stability",
        "self_consistency",
        "semantic_entropy",
        "model_disagreement",
        "calibrated_probability",
        "answer_logit_margin",
        "answer_logit_execute_probability",
        "answer_logit_confidence",
        "answer_logit_uncertainty",
        "evidence_support",
        "evidence_conflict",
        "source_reliability",
        "tool_agreement",
        "ood_score",
        "source_risk",
    }
)


@dataclass(frozen=True)
class RetrievalResult:
    record_id: str
    distance: float
    outcome_success: int
    observed_utility: float
    candidate_action: str

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "distance": self.distance,
            "outcome_success": self.outcome_success,
            "observed_utility": self.observed_utility,
            "candidate_action": self.candidate_action,
        }


class HistoricalEvidenceRetriever:
    """Label-safe for the target action: only outcomes of train-library analogs are exposed."""

    def __init__(self, train_records: list[ActionRecord], *, neighbors: int = 5) -> None:
        if not train_records:
            raise ValueError("retrieval library cannot be empty")
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.train_records = list(train_records)
        self.neighbors = neighbors
        self.feature_names = [
            name
            for name in feature_names_from_records(train_records)
            if name not in EXCLUDED_RETRIEVAL_FEATURES and not name.startswith("task::")
        ]
        rows = [self._row(record) for record in self.train_records]
        self.standardizer = Standardizer().fit(rows)
        self.standardized_rows = self.standardizer.transform(rows)
        self.task_indices = {
            task: [index for index, record in enumerate(self.train_records) if record.benchmark == task]
            for task in sorted({record.benchmark for record in self.train_records})
        }

    def evidence_views(self, record: ActionRecord) -> list[EvidenceView]:
        if record.record_id in {candidate.record_id for candidate in self.train_records}:
            raise ValueError("target action must not appear in the retrieval library")
        analogs = self.retrieve(record)
        analog_labels = [result.outcome_success for result in analogs]
        agreement = abs(mean(analog_labels) - 0.5) * 2.0 if analog_labels else 0.0
        mean_distance = mean(result.distance for result in analogs) if analogs else 1.0
        analog_features = dict(record.features)
        analog_features.update(
            {
                "evidence_support": mean(analog_labels) if analog_labels else 0.0,
                "evidence_conflict": 1.0 - agreement,
                "tool_agreement": agreement,
                "source_reliability": 0.9,
                "ood_score": min(1.0, mean_distance / 4.0),
                "source_risk": 0.1,
            }
        )
        analog_lines = [
            "Historical nearest-neighbor retrieval from the training library (target outcome excluded):"
        ]
        analog_lines.extend(
            f"- {result.candidate_action}; historical_success={result.outcome_success}; "
            f"historical_utility={result.observed_utility:.4f}; standardized_distance={result.distance:.4f}"
            for result in analogs
        )
        analog_view = EvidenceView(
            view_id="tool::nearest_analogs",
            record=replace(
                record,
                evidence=[*record.evidence, *analog_lines],
                features=analog_features,
            ),
            intervention="historical_nearest_neighbor_retrieval",
            provenance={
                "type": "offline_training_library_tool",
                "label_blind_for_target": True,
                "results": [result.to_json() for result in analogs],
            },
        )

        task_records = [self.train_records[index] for index in self.task_indices[record.benchmark]]
        success_rate = mean(candidate.label for candidate in task_records)
        mean_utility = mean(candidate.utility for candidate in task_records)
        task_features = dict(record.features)
        task_features.update(
            {
                "evidence_support": success_rate,
                "evidence_conflict": 2.0 * min(success_rate, 1.0 - success_rate),
                "tool_agreement": abs(success_rate - 0.5) * 2.0,
                "source_reliability": 0.95,
                "source_risk": 0.05,
            }
        )
        task_view = EvidenceView(
            view_id="tool::task_statistics",
            record=replace(
                record,
                evidence=[
                    *record.evidence,
                    "Historical task-level training-library statistics (target outcome excluded):",
                    f"- n={len(task_records)}; success_rate={success_rate:.4f}; mean_observed_utility={mean_utility:.4f}",
                ],
                features=task_features,
            ),
            intervention="historical_task_statistics",
            provenance={
                "type": "offline_training_library_tool",
                "label_blind_for_target": True,
                "n": len(task_records),
                "success_rate": success_rate,
                "mean_observed_utility": mean_utility,
            },
        )
        base_view = EvidenceView(
            view_id="tool::base_evidence",
            record=record,
            intervention="none",
            provenance={"type": "observed_record", "label_blind_for_target": True},
        )
        return [base_view, analog_view, task_view]

    def retrieve(self, record: ActionRecord) -> list[RetrievalResult]:
        target = self.standardizer.transform([self._row(record)])[0]
        candidates = []
        raw_id = str(record.metadata.get("raw_record_id", record.record_id))
        for index in self.task_indices.get(record.benchmark, []):
            candidate = self.train_records[index]
            candidate_raw_id = str(candidate.metadata.get("raw_record_id", candidate.record_id))
            if candidate.record_id == record.record_id or candidate_raw_id == raw_id:
                continue
            distance = math.sqrt(
                sum((left - right) ** 2 for left, right in zip(target, self.standardized_rows[index]))
            )
            candidates.append((distance, candidate.record_id, candidate))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [
            RetrievalResult(
                record_id=candidate.record_id,
                distance=distance,
                outcome_success=int(candidate.label),
                observed_utility=_retrieval_utility(candidate),
                candidate_action=candidate.candidate_action,
            )
            for distance, _, candidate in candidates[: self.neighbors]
        ]

    def _row(self, record: ActionRecord) -> list[float]:
        return [float(record.features.get(name, 0.0)) for name in self.feature_names]


def _retrieval_utility(record: ActionRecord) -> float:
    hidden = record.metadata.get("hidden_outcome_for_evaluation_only")
    if isinstance(hidden, dict) and "utility_difference" in hidden:
        return float(hidden["utility_difference"])
    return float(record.utility)
