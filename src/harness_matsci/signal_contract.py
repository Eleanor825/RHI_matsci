from __future__ import annotations

from typing import Any

from .schema import ActionRecord


INTERNAL_UNCERTAINTY_SIGNALS = (
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
)

EXTERNAL_UNCERTAINTY_SIGNALS = (
    "evidence_support",
    "evidence_conflict",
    "source_reliability",
    "tool_agreement",
    "ood_score",
    "source_risk",
)

ALL_UNCERTAINTY_SIGNALS = INTERNAL_UNCERTAINTY_SIGNALS + EXTERNAL_UNCERTAINTY_SIGNALS


def signal_coverage(records: list[ActionRecord]) -> dict[str, dict[str, Any]]:
    total = len(records)
    return {
        name: {
            "count": sum(name in record.features for record in records),
            "fraction": (
                sum(name in record.features for record in records) / total if total else 0.0
            ),
        }
        for name in ALL_UNCERTAINTY_SIGNALS
    }


def require_signal_contract(
    records: list[ActionRecord],
    *,
    required: tuple[str, ...] = ALL_UNCERTAINTY_SIGNALS,
    min_fraction: float = 1.0,
) -> None:
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_fraction must be in (0, 1]")
    coverage = signal_coverage(records)
    missing = [
        name for name in required if coverage[name]["fraction"] < min_fraction
    ]
    if missing:
        details = ", ".join(
            f"{name}={coverage[name]['count']}/{len(records)}" for name in missing
        )
        raise ValueError(f"uncertainty signal contract is incomplete: {details}")
