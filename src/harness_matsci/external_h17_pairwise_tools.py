from __future__ import annotations

from dataclasses import replace

from .external_h16_pairwise_tools import ExternalH16PairwiseEvidenceEnvironment


EVIDENCE_SOURCES = ("agent_prior", "composition", "structure", "high_fidelity")


class ExternalH17PairwiseEvidenceEnvironment(ExternalH16PairwiseEvidenceEnvironment):
    def observe_many(self, records, *, source):
        if source != "agent_prior":
            return super().observe_many(records, source=source)
        return [self._agent_prior_view(record) for record in records]

    def _agent_prior_view(self, record):
        return replace(
            record,
            evidence=[
                *record.evidence,
                f"agent prior margin={record.features['agent_prior_margin']:.6f}",
                (
                    "agent prior estimated pair utility="
                    f"{record.features['agent_prior_estimated_utility']:.6f}"
                ),
            ],
            features={
                **record.features,
                "tool_estimated_margin": record.features["agent_prior_margin"],
                "tool_uncertainty": record.features["agent_prior_uncertainty"],
                "tool_source_confidence": record.features["agent_prior_confidence"],
                "tool_source_confidence_low": record.features[
                    "agent_prior_confidence_low"
                ],
                "tool_source_confidence_high": record.features[
                    "agent_prior_confidence_high"
                ],
                "tool_estimated_utility": record.features[
                    "agent_prior_estimated_utility"
                ],
                "tool_estimated_utility_low": record.features[
                    "agent_prior_estimated_utility_low"
                ],
                "tool_estimated_utility_high": record.features[
                    "agent_prior_estimated_utility_high"
                ],
                "tool_source_agent_prior": 1.0,
                "tool_source_composition": 0.0,
                "tool_source_structure": 0.0,
                "tool_source_high_fidelity": 0.0,
                "tool_cost": 0.0,
            },
            metadata={
                **record.metadata,
                "evidence_source": "agent_prior",
                "evidence_provenance": {
                    "source_family": "agent_internal",
                    "tool": "train_only_agent_ensemble",
                    "target_label_exposed": False,
                },
                "hidden_outcome_exposed": False,
                "outcome_defining_measurement_replayed": False,
            },
        )
