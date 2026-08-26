from __future__ import annotations

from dataclasses import replace

from .replica_contract import EvidenceView
from .retrieval_views import HistoricalEvidenceRetriever
from .schema import ActionRecord


class LfpChoiceEvidenceViewBuilder:
    def __init__(
        self,
        train_records: list[ActionRecord],
        *,
        neighbors: int = 5,
    ) -> None:
        self.retriever = HistoricalEvidenceRetriever(
            train_records,
            neighbors=min(neighbors, len(train_records)),
        )

    def views(self, record: ActionRecord) -> list[EvidenceView]:
        retrieved = self.retriever.evidence_views(record)
        analog = next(view for view in retrieved if view.view_id == "tool::nearest_analogs")
        base = EvidenceView(
            view_id="evidence::base",
            record=replace(record, record_id=f"{record.record_id}::view-base"),
            intervention="none",
            provenance={"type": "observed_pre_action_record", "target_outcome_hidden": True},
        )
        directional = sorted(
            (
                name.removeprefix("candidate_delta::"),
                float(value),
            )
            for name, value in record.features.items()
            if name.startswith("candidate_delta::") and abs(float(value)) > 1e-12
        )
        descriptor_lines = [
            "Deterministic formulation-difference tool output; positive means candidate A exceeds candidate B:"
        ]
        descriptor_lines.extend(
            f"- {name}={value:+.6f}" for name, value in directional
        )
        descriptor_lines.extend(
            [
                f"- visible_feature_distance={record.features['candidate_visible_feature_distance']:.6f}",
                "- target performance outcomes remain withheld",
            ]
        )
        descriptor = EvidenceView(
            view_id="tool::formulation_descriptors",
            record=replace(
                record,
                record_id=f"{record.record_id}::view-descriptors",
                evidence=[*record.evidence, *descriptor_lines],
            ),
            intervention="deterministic_outcome_blind_descriptor_tool",
            provenance={
                "type": "deterministic_tool",
                "target_outcome_hidden": True,
                "directional_feature_count": len(directional),
            },
        )
        analog_record = replace(
            analog.record,
            record_id=f"{record.record_id}::view-analogs",
        )
        analog_view = EvidenceView(
            view_id="tool::held_batch_analogs",
            record=analog_record,
            intervention=analog.intervention,
            provenance={
                **analog.provenance,
                "target_outcome_hidden": True,
                "retrieval_library_excludes_target_batch": True,
            },
        )
        return [base, descriptor, analog_view]
