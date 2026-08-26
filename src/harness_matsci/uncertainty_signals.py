from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .logit_uncertainty import execute_probability
from .schema import ActionRecord


@dataclass(frozen=True)
class UncertaintySignal:
    record_id: str
    confidence: float
    uncertainty: float
    source: str
    metadata: dict[str, Any]
    features: dict[str, float] = field(default_factory=dict)

    def to_features(self) -> dict[str, float]:
        return {
            "llm_signal_confidence": float(self.confidence),
            "llm_signal_uncertainty": float(self.uncertainty),
            **{str(key): float(value) for key, value in self.features.items()},
        }


class UncertaintySignalProvider(Protocol):
    name: str

    def score(self, records: list[ActionRecord]) -> dict[str, UncertaintySignal]:
        ...


class CachedConfidenceProvider:
    """Adapter for GPT/API self-reports or cached judge scores.

    Values must be produced at decision time and calibrated only on the
    training/feedback side of the experiment. This adapter deliberately does
    not imply that the score is an intrinsic model uncertainty.
    """

    name = "cached_self_report"

    def __init__(self, scores: dict[str, float], *, source: str = "llm_self_report") -> None:
        self.scores = {str(key): max(0.0, min(1.0, float(value))) for key, value in scores.items()}
        self.source = source

    def score(self, records: list[ActionRecord]) -> dict[str, UncertaintySignal]:
        missing = [record.record_id for record in records if record.record_id not in self.scores]
        if missing:
            raise ValueError(f"missing uncertainty scores for {len(missing)} records; first={missing[0]}")
        return {
            record.record_id: UncertaintySignal(
                record_id=record.record_id,
                confidence=self.scores[record.record_id],
                uncertainty=1.0 - self.scores[record.record_id],
                source=self.source,
                metadata={},
            )
            for record in records
        }


class DirectJudgeSignalProvider:
    """Adapter for any judge exposing ``score_records(records)``."""

    name = "direct_llm_judge"

    def __init__(self, judge: Any, *, source: str = "llm_self_report") -> None:
        if not hasattr(judge, "score_records"):
            raise TypeError("judge must expose score_records(records)")
        self.judge = judge
        self.source = source

    def score(self, records: list[ActionRecord]) -> dict[str, UncertaintySignal]:
        scores = self.judge.score_records(records)
        return CachedConfidenceProvider(scores, source=self.source).score(records)


class AnswerLogitSignalProvider:
    """Decision-time answer-logit difference provider.

    The prompt is converted to a binary next-token decision: token A means
    execute the candidate and token B means defer execution. This follows the
    answer-logit uncertainty construction and requires a judge exposing
    ``score(prompts)`` with margin, confidence, uncertainty, and preferred.
    """

    name = "answer_logit_difference"

    def __init__(
        self,
        judge: Any,
        *,
        prompt_builder: Callable[[ActionRecord], str] | None = None,
        source: str = "answer_logit_difference",
    ) -> None:
        if not hasattr(judge, "score"):
            raise TypeError("judge must expose score(prompts)")
        self.judge = judge
        self.prompt_builder = prompt_builder or _answer_logit_prompt
        self.source = source

    def score(self, records: list[ActionRecord]) -> dict[str, UncertaintySignal]:
        margins = self.judge.score([self.prompt_builder(record) for record in records])
        if len(margins) != len(records):
            raise ValueError("answer-logit judge returned the wrong number of scores")
        output: dict[str, UncertaintySignal] = {}
        for record, margin in zip(records, margins):
            probability = float(getattr(margin, "execute_probability", execute_probability(float(margin.margin))))
            output[record.record_id] = UncertaintySignal(
                record_id=record.record_id,
                confidence=float(margin.confidence),
                uncertainty=float(margin.uncertainty),
                source=self.source,
                metadata={"preferred": str(margin.preferred)},
                features={
                    "answer_logit_margin": float(margin.margin),
                    "answer_logit_execute_probability": probability,
                    "answer_logit_confidence": float(margin.confidence),
                    "answer_logit_uncertainty": float(margin.uncertainty),
                },
            )
        return output


class CompositeSignalProvider:
    """Merge multiple signal sources without hard-coding model availability."""

    name = "composite"

    def __init__(self, providers: list[UncertaintySignalProvider]) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = list(providers)

    def score(self, records: list[ActionRecord]) -> dict[str, UncertaintySignal]:
        outputs = [provider.score(records) for provider in self.providers]
        merged: dict[str, UncertaintySignal] = {}
        for record in records:
            signals = [output[record.record_id] for output in outputs]
            confidence = sum(signal.confidence for signal in signals) / len(signals)
            uncertainty = sum(signal.uncertainty for signal in signals) / len(signals)
            merged[record.record_id] = UncertaintySignal(
                record_id=record.record_id,
                confidence=confidence,
                uncertainty=uncertainty,
                source="+".join(signal.source for signal in signals),
                metadata={"sources": [signal.source for signal in signals]},
                features={
                    key: value
                    for signal in signals
                    for key, value in signal.features.items()
                },
            )
        return merged


def attach_signal_features(
    records: list[ActionRecord],
    provider: UncertaintySignalProvider,
) -> list[ActionRecord]:
    signals = provider.score(records)
    return [
        ActionRecord(
            record_id=record.record_id,
            benchmark=record.benchmark,
            split=record.split,
            visible_context=record.visible_context,
            candidate_action=record.candidate_action,
            action_type=record.action_type,
            evidence=list(record.evidence),
            features={**record.features, **signals[record.record_id].to_features()},
            label=record.label,
            utility=record.utility,
            metadata={**record.metadata, "uncertainty_signal_source": signals[record.record_id].source},
        )
        for record in records
    ]


def _answer_logit_prompt(record: ActionRecord) -> str:
    evidence = "\n".join(f"- {item}" for item in record.evidence) or "- none"
    return (
        "Decide whether the proposed scientific action is worth executing now.\n"
        "Use only the visible context and evidence.\n\n"
        f"Context:\n{record.visible_context}\n\n"
        f"Proposed action:\n{record.candidate_action}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer with exactly one token: A if execute now, B if defer or abstain.\n"
        "Answer:"
    )
