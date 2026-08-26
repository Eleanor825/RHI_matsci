from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .features import DEFAULT_FEATURES
from .harnesses import DEFAULT_HARNESS
from .metrics import action_worthiness_score
from .schema import ActionRecord
from .signal_contract import EXTERNAL_UNCERTAINTY_SIGNALS, INTERNAL_UNCERTAINTY_SIGNALS
from .training import TrainedGate, evaluate_gate, train_gate_with_features


@dataclass(frozen=True)
class NumericCritic:
    gate: TrainedGate
    utility_model: Any
    feature_names: tuple[str, ...]
    utility_fallback: float

    def predict_utility(self, records: list[ActionRecord]) -> list[float]:
        if self.utility_model is None:
            return [self.utility_fallback] * len(records)
        matrix = [[float(record.features.get(name, 0.0)) for name in self.feature_names] for record in records]
        return [min(1.0, max(0.0, float(value))) for value in self.utility_model.predict(matrix)]


@dataclass(frozen=True)
class _ConstantUtilityModel:
    value: float

    def predict(self, matrix: list[list[float]]) -> list[float]:
        return [self.value] * len(matrix)


def fit_numeric_critic(
    train_records: list[ActionRecord],
    calibration_records: list[ActionRecord],
    *,
    feature_names: tuple[str, ...],
    alpha: float,
    epochs: int,
) -> NumericCritic:
    gate = train_gate_with_features(
        train_records,
        calibration_records,
        feature_names=list(feature_names),
        alpha=alpha,
        epochs=epochs,
    )
    successful = [record for record in train_records if record.label == 1]
    fallback = sum((min(1.0, max(0.0, record.utility)) for record in successful), 0.0) / len(successful) if successful else 0.0
    utility_model: Any = _ConstantUtilityModel(fallback)
    if len(successful) >= 10:
        try:
            from sklearn.ensemble import ExtraTreesRegressor
        except ModuleNotFoundError:
            pass
        else:
            matrix = [[float(record.features.get(name, 0.0)) for name in feature_names] for record in successful]
            targets = [min(1.0, max(0.0, float(record.utility))) for record in successful]
            utility_model = ExtraTreesRegressor(
                n_estimators=80,
                min_samples_leaf=max(2, len(successful) // 40),
                random_state=17,
                n_jobs=1,
            )
            utility_model.fit(matrix, targets)
    return NumericCritic(gate, utility_model, feature_names, fallback)


@dataclass(frozen=True)
class ActionFeedback:
    record_id: str
    benchmark: str
    predicted_probability: float
    label: int
    residual: float
    confidently_wrong: bool
    internal_signal_values: dict[str, float]
    external_signal_values: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessDefect:
    defect_type: str
    severity: float
    evidence_count: int
    affected_signals: tuple[str, ...]
    diagnosis: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def score_action_records(records: list[ActionRecord], critic: NumericCritic | TrainedGate) -> list[dict[str, Any]]:
    """Export the runtime numeric critic output without oracle fields."""
    gate = critic.gate if isinstance(critic, NumericCritic) else critic
    probabilities = gate.predict_proba(records)
    utilities = critic.predict_utility(records) if isinstance(critic, NumericCritic) else [1.0] * len(records)
    scored = []
    for record, probability, utility in zip(records, probabilities, utilities):
        p_success = min(1.0 - 1e-9, max(1e-9, float(probability)))
        entropy = -(p_success * math.log(p_success) + (1.0 - p_success) * math.log(1.0 - p_success)) / math.log(2.0)
        cost = float(record.features.get("cost", 0.0))
        reversibility = float(record.features.get("reversibility", 0.5))
        failure_harm = 0.5 * cost + 0.5 * (1.0 - reversibility)
        expected_net_gain = p_success * utility - 0.15 * cost - 0.25 * (1.0 - p_success) * failure_harm
        scored.append({
            "record_id": record.record_id,
            "benchmark": record.benchmark,
            "p_success": p_success,
            "uncertainty": entropy,
            "utility_if_success": utility,
            "expected_net_gain": expected_net_gain,
            "cost": cost,
            "failure_harm": failure_harm,
            "threshold": float(gate.threshold),
        })
    return scored


@dataclass(frozen=True)
class EvolvingHarnessRevision:
    round_index: int
    name: str
    feature_names: tuple[str, ...]
    target_selective_risk: float
    defect: HarnessDefect
    proposal_reason: str
    accepted: bool
    acceptance_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _split_four_way(records: list[ActionRecord], seed: int) -> tuple[list[ActionRecord], ...]:
    if len(records) < 20:
        raise ValueError("evolving harness needs at least twenty records")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    cuts = (int(n * 0.60), int(n * 0.75), int(n * 0.90))
    return shuffled[: cuts[0]], shuffled[cuts[0] : cuts[1]], shuffled[cuts[1] : cuts[2]], shuffled[cuts[2] :]


def _available_features(records: Iterable[ActionRecord]) -> tuple[str, ...]:
    names = {name for record in records for name in record.features}
    required = set(INTERNAL_UNCERTAINTY_SIGNALS) | set(EXTERNAL_UNCERTAINTY_SIGNALS)
    present = names & required
    if not present:
        raise ValueError("records contain no internal or external uncertainty signals")
    return tuple(sorted(present))


def _prediction_feedback(records: list[ActionRecord], gate: TrainedGate) -> list[ActionFeedback]:
    probabilities = gate.predict_proba(records)
    internal = set(INTERNAL_UNCERTAINTY_SIGNALS)
    external = set(EXTERNAL_UNCERTAINTY_SIGNALS)
    feedback = []
    for record, probability in zip(records, probabilities):
        residual = abs(float(probability) - float(record.label))
        feedback.append(
            ActionFeedback(
                record_id=record.record_id,
                benchmark=record.benchmark,
                predicted_probability=float(probability),
                label=int(record.label),
                residual=residual,
                confidently_wrong=bool(probability >= gate.threshold and record.label == 0),
                internal_signal_values={k: float(v) for k, v in record.features.items() if k in internal},
                external_signal_values={k: float(v) for k, v in record.features.items() if k in external},
            )
        )
    return feedback


def _signal_error_association(feedback: list[ActionFeedback]) -> dict[str, float]:
    if not feedback:
        return {}
    associations: dict[str, float] = {}
    for group in ("internal_signal_values", "external_signal_values"):
        names = sorted({name for row in feedback for name in getattr(row, group)})
        for name in names:
            pairs = [(getattr(row, group).get(name), row.residual) for row in feedback if name in getattr(row, group)]
            if len(pairs) < 2:
                continue
            x_mean = sum(x for x, _ in pairs) / len(pairs)
            y_mean = sum(y for _, y in pairs) / len(pairs)
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
            denominator = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs) * sum((y - y_mean) ** 2 for _, y in pairs))
            associations[name] = abs(numerator / denominator) if denominator > 1e-12 else 0.0
    return associations


def _diagnose(feedback: list[ActionFeedback], gate: TrainedGate) -> HarnessDefect:
    if not feedback:
        return HarnessDefect("no_feedback", 0.0, 0, (), "No action-level feedback was available.")
    wrong = sum(row.confidently_wrong for row in feedback)
    mean_residual = sum(row.residual for row in feedback) / len(feedback)
    associations = _signal_error_association(feedback)
    affected = tuple(name for name, _ in sorted(associations.items(), key=lambda item: (-item[1], item[0]))[:4])
    if wrong / len(feedback) > 0.10:
        return HarnessDefect(
            "overconfident_failure",
            min(1.0, wrong / len(feedback) * 2.0),
            wrong,
            affected,
            "The active contract accepts too many failed actions at high predicted probability.",
        )
    if mean_residual > 0.30:
        return HarnessDefect(
            "miscalibration",
            min(1.0, mean_residual),
            len(feedback),
            affected,
            "Predicted action reliability is poorly aligned with observed action outcomes.",
        )
    return HarnessDefect(
        "stable_contract",
        max(0.0, mean_residual),
        len(feedback),
        affected,
        "No dominant action-level failure mode exceeded the mutation threshold.",
    )


def _candidate_features(current: tuple[str, ...], all_features: tuple[str, ...], defect: HarnessDefect) -> tuple[str, ...]:
    selected = list(current)
    ranked = list(defect.affected_signals)
    if defect.defect_type in {"overconfident_failure", "miscalibration"}:
        for name in ranked:
            if name not in selected:
                selected.append(name)
        # A high-risk contract should not rely on verbal confidence alone.
        if "verbal_confidence" in selected and "model_disagreement" in all_features:
            selected.remove("verbal_confidence")
            if "model_disagreement" not in selected:
                selected.append("model_disagreement")
    return tuple(sorted(dict.fromkeys(selected)))


def _revision_score(report: dict[str, Any]) -> float:
    return action_worthiness_score(report.get("metrics", {}), report.get("discovery_gain", {}))


def run_evolving_harness(
    records: list[ActionRecord],
    *,
    iterations: int = 3,
    seed: int = 7,
    alpha: float = 0.10,
    budget_fraction: float = 0.10,
    epochs: int = 300,
    epsilon: float = 0.005,
) -> dict[str, Any]:
    """Run numeric-critic RHI with action feedback and held-out acceptance.

    The actor remains outside this module. The harness emits calibrated numeric
    action scores; feedback diagnoses systematic errors, proposes a new feature
    contract, and accepts it only on an untouched acceptance split.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    train, feedback_records, acceptance, test = _split_four_way(records, seed)
    all_features = _available_features(records)
    current_features = tuple(name for name in DEFAULT_FEATURES if name in all_features)
    if not current_features:
        current_features = all_features
    current_harness = copy.deepcopy(DEFAULT_HARNESS)
    current_harness.update({"name": "H0_numeric_critic", "required_features": list(current_features), "target_selective_risk": alpha})
    current_critic = fit_numeric_critic(train, feedback_records, feature_names=current_features, alpha=alpha, epochs=epochs)
    current_gate = current_critic.gate
    incumbent_acceptance = evaluate_gate(acceptance, current_gate, budget_fraction=budget_fraction)
    revisions: list[EvolvingHarnessRevision] = []
    trace: list[dict[str, Any]] = []
    for round_index in range(1, iterations + 1):
        action_feedback = _prediction_feedback(feedback_records, current_gate)
        defect = _diagnose(action_feedback, current_gate)
        proposed_features = _candidate_features(current_features, all_features, defect)
        candidate_critic = fit_numeric_critic(train, feedback_records, feature_names=proposed_features, alpha=alpha, epochs=epochs)
        candidate_gate = candidate_critic.gate
        candidate_acceptance = evaluate_gate(acceptance, candidate_gate, budget_fraction=budget_fraction)
        current_score = _revision_score(incumbent_acceptance)
        candidate_score = _revision_score(candidate_acceptance)
        candidate_risk = float(candidate_acceptance["metrics"].get("selective_risk", 1.0))
        accepted = candidate_risk <= alpha and candidate_score <= current_score - epsilon
        proposal_reason = f"{defect.defect_type}: {defect.diagnosis}"
        revision = EvolvingHarnessRevision(round_index, f"H{round_index}_numeric_critic", proposed_features, alpha, defect, proposal_reason, accepted, candidate_acceptance)
        revisions.append(revision)
        trace.append({"round": round_index, "defect": defect.to_json(), "candidate_features": list(proposed_features), "incumbent_score": current_score, "candidate_score": candidate_score, "accepted": accepted})
        if accepted:
            current_features = proposed_features
            current_gate = candidate_gate
            current_critic = candidate_critic
            current_harness = copy.deepcopy(current_harness)
            current_harness.update({"name": revision.name, "required_features": list(current_features), "evolution_reason": proposal_reason})
            incumbent_acceptance = candidate_acceptance
    final_test = evaluate_gate(test, current_gate, budget_fraction=budget_fraction)
    action_scores = score_action_records(test, current_critic)
    revisions_with_test = []
    for revision in revisions:
        revisions_with_test.append(revision.to_json())
    return {
        "method": {
            "name": "EVL-Harness: action-feedback numeric critic with recursive harness evolution",
            "actor_harness_interface": "numeric-only action score; the MatBot actor chooses the next action",
            "feedback_loop": ["score action", "observe hidden outcome", "diagnose harness defect", "propose feature-contract mutation", "accept on held-out shard"],
            "internal_signals": [name for name in current_features if name in INTERNAL_UNCERTAINTY_SIGNALS],
            "external_signals": [name for name in current_features if name in EXTERNAL_UNCERTAINTY_SIGNALS],
            "test_is_used_only_after_selection": True,
        },
        "config": {"iterations": iterations, "seed": seed, "alpha": alpha, "budget_fraction": budget_fraction, "epsilon": epsilon},
        "sizes": {"train": len(train), "feedback": len(feedback_records), "acceptance": len(acceptance), "test": len(test)},
        "initial_harness": {"name": "H0_numeric_critic", "required_features": list(current_features), "target_selective_risk": alpha},
        "final_harness": current_harness,
        "action_scores": action_scores,
        "revisions": revisions_with_test,
        "trace": trace,
        "feedback_summary": {
            "n_feedback_actions": len(feedback_records),
            "mean_absolute_residual": sum(row.residual for row in _prediction_feedback(feedback_records, current_gate)) / len(feedback_records),
            "confidently_wrong_count": sum(row.confidently_wrong for row in _prediction_feedback(feedback_records, current_gate)),
        },
        "test": final_test,
    }
