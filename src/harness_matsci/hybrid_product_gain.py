from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .conformal_gain import ConformalGainObservation, fit_split_conformal_gain
from .continuous_gain_calibration import (
    ContinuousGainObservation,
    fit_continuous_gain_calibrator,
)
from .metrics import binary_metrics
from .llm_action_choice import PROMPT_VERSION
from .product_gain_decomposition import decompose_product_gain
from .risk_control import binomial_upper_confidence
from .schema import ActionRecord


HYBRID_COST = 0.01


@dataclass(frozen=True)
class HybridProductState:
    record_id: str
    benchmark: str
    objective: str
    proposal_sign: float
    model_probabilities_a: tuple[float, ...]
    success_probabilities: tuple[float, ...]
    mean_success_probability: float
    mean_value_magnitude: float
    predicted_mean_net_gain: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    actual_net_gain: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def build_hybrid_product_states(
    records: list[ActionRecord],
    raw_states: list[dict[str, object]],
    llm_probabilities_a: dict[str, tuple[float, ...]],
    source_weights: dict[str, float],
) -> list[HybridProductState]:
    raw_by_id = {str(state["record_id"]): state for state in raw_states}
    source_order = ("full", "solvent_salt", "additive", "historical_analog")
    output = []
    for record in records:
        probabilities_a = tuple(llm_probabilities_a[record.record_id])
        proposal_sign = 1.0 if sum(probabilities_a) / len(probabilities_a) >= 0.5 else -1.0
        success_probabilities = tuple(
            probability if proposal_sign > 0.0 else 1.0 - probability
            for probability in probabilities_a
        )
        raw = raw_by_id[record.record_id]
        source_matrix = tuple(tuple(float(value) for value in row) for row in raw["source_matrix"])
        magnitudes = tuple(abs(value) for row in source_matrix for value in row)
        magnitude_weights = tuple(
            source_weights[source] / len(source_matrix)
            for _ in source_matrix
            for source in source_order
        )
        decomposition = decompose_product_gain(
            success_probabilities,
            magnitudes,
            value_weights=magnitude_weights,
            cost=HYBRID_COST,
        )
        utility_difference = float(
            record.metadata["hidden_outcome_for_evaluation_only"]["utility_difference"]
        )
        output.append(
            HybridProductState(
                record_id=record.record_id,
                benchmark=record.benchmark,
                objective=str(record.metadata["objective"]),
                proposal_sign=proposal_sign,
                model_probabilities_a=probabilities_a,
                success_probabilities=success_probabilities,
                mean_success_probability=decomposition.mean_success_probability,
                mean_value_magnitude=decomposition.mean_value_magnitude,
                predicted_mean_net_gain=decomposition.mean_net_gain,
                internal_variance=decomposition.internal_variance,
                external_variance=decomposition.external_variance,
                interaction_variance=decomposition.interaction_variance,
                actual_net_gain=proposal_sign * utility_difference - HYBRID_COST,
            )
        )
    return output


def load_llm_probability_caches(
    records: list[ActionRecord],
    *,
    model_cache_paths: dict[str, str | Path],
    replica_id: int = 0,
) -> dict[str, tuple[float, ...]]:
    """Load a complete frozen model grid without silently dropping records."""
    predictions_by_model = {}
    for model, path in model_cache_paths.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != model:
            raise ValueError(f"cache model mismatch for {model}")
        if payload.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"cache prompt version mismatch for {model}")
        predictions_by_model[model] = payload.get("predictions", {})
    output = {}
    for record in records:
        key = f"{PROMPT_VERSION}::{record.record_id}::replica-{replica_id}"
        probabilities = []
        for model in model_cache_paths:
            prediction = predictions_by_model[model].get(key)
            if prediction is None:
                raise ValueError(f"missing frozen prediction: model={model}, record={record.record_id}")
            probabilities.append(float(prediction["p_a_better"]))
        output[record.record_id] = tuple(probabilities)
    return output


def evaluate_hybrid_product_gain(
    *,
    feedback: list[HybridProductState],
    acceptance: list[HybridProductState],
    test: list[HybridProductState],
    model_names: tuple[str, ...] | None = None,
    alpha: float = 0.10,
    delta: float = 0.05,
) -> dict[str, object]:
    calibrator = fit_continuous_gain_calibrator(
        [_observation(state) for state in feedback],
        use_source_variance=True,
    )
    conformal_states, certificate_states = _split_acceptance(acceptance)
    conformal_predictions = calibrator.predict(
        [_observation(state) for state in conformal_states]
    )
    conformal = fit_split_conformal_gain(
        [
            ConformalGainObservation(
                record_id=state.record_id,
                predicted_mean_gain=prediction.calibrated_mean_gain,
                predicted_scale=math.sqrt(prediction.calibrated_variance),
                observed_gain=state.actual_net_gain,
            )
            for state, prediction in zip(conformal_states, conformal_predictions)
        ],
        alpha=alpha,
    )
    certificate_rows = _predict(certificate_states, calibrator, conformal)
    selected_certificate = [
        row for row in certificate_rows if row["lower_gain_bound"] > 0.0
    ]
    errors = sum(row["actual_net_gain"] <= 0.0 for row in selected_certificate)
    upper = binomial_upper_confidence(errors, len(selected_certificate), delta)
    certified = bool(selected_certificate) and upper <= alpha
    test_rows = _predict(test, calibrator, conformal)
    deployed = (
        [row for row in test_rows if row["lower_gain_bound"] > 0.0]
        if certified
        else []
    )
    labels = [int(row["actual_net_gain"] > 0.0) for row in test_rows]
    probabilities = [row["action_worthiness"] for row in test_rows]
    direct_probabilities = [state.mean_success_probability for state in test]
    if model_names is None:
        model_names = tuple(f"model_{index}" for index in range(len(test[0].model_probabilities_a)))
    if test and len(model_names) != len(test[0].model_probabilities_a):
        raise ValueError("model names must align with model probability samples")
    return {
        "calibrator": calibrator.to_json(),
        "conformal": conformal.to_json(),
        "certificate": {
            "policy": "lower_gain_bound > 0",
            "selected_count": len(selected_certificate),
            "errors": errors,
            "empirical_risk": errors / len(selected_certificate) if selected_certificate else 0.0,
            "risk_upper_bound": upper,
            "alpha": alpha,
            "delta": delta,
            "certified": certified,
        },
        "hybrid_metrics": binary_metrics(labels, probabilities, threshold=0.5),
        "direct_llm_ensemble_metrics": binary_metrics(
            labels, direct_probabilities, threshold=0.5
        ),
        "direct_llm_model_metrics": {
            model: _direct_model_metrics(test, model_index)
            for model_index, model in enumerate(model_names)
        },
        "component_ablation_metrics": {
            mode: _component_ablation_metrics(feedback, test, mode)
            for mode in ("mean_only", "internal_only", "internal_external", "full")
        },
        "test_conformal_coverage": _mean(
            float(row["actual_net_gain"] >= row["lower_gain_bound"])
            for row in test_rows
        ),
        "deployed": {
            "selected_count": len(deployed),
            "coverage": len(deployed) / len(test_rows) if test_rows else 0.0,
            "selective_risk": _mean(
                float(row["actual_net_gain"] <= 0.0) for row in deployed
            ),
            "population_net_gain": sum(row["actual_net_gain"] for row in deployed)
            / len(test_rows)
            if test_rows
            else 0.0,
            "selected_mean_net_gain": _mean(
                row["actual_net_gain"] for row in deployed
            ),
        },
        "test_rows": test_rows,
    }


def _observation(state: HybridProductState) -> ContinuousGainObservation:
    return ContinuousGainObservation(
        record_id=state.record_id,
        benchmark="lfp_real_hybrid_product_gain",
        mean_gain=state.predicted_mean_net_gain,
        internal_variance=state.internal_variance,
        external_variance=state.external_variance,
        interaction_variance=state.interaction_variance,
        gain=state.actual_net_gain,
    )


def _predict(states, calibrator, conformal):
    predictions = calibrator.predict([_observation(state) for state in states])
    output = []
    for state, prediction in zip(states, predictions):
        bound = conformal.predict_one(
            record_id=state.record_id,
            predicted_mean_gain=prediction.calibrated_mean_gain,
            predicted_scale=math.sqrt(prediction.calibrated_variance),
        )
        output.append(
            {
                **state.to_json(),
                "calibrated_mean_gain": prediction.calibrated_mean_gain,
                "calibrated_variance": prediction.calibrated_variance,
                "action_worthiness": prediction.action_worthiness,
                "lower_gain_bound": bound.lower_gain_bound,
            }
        )
    return output


def _split_acceptance(states):
    left = []
    right = []
    for state in states:
        parity = int(hashlib.sha256(state.record_id.encode()).hexdigest()[-1], 16) % 2
        (left if parity == 0 else right).append(state)
    if not left or not right:
        raise ValueError("acceptance must populate conformal and certificate splits")
    return left, right


def _direct_model_metrics(
    states: list[HybridProductState], model_index: int
) -> dict[str, float]:
    labels = []
    probabilities = []
    for state in states:
        probability_a = state.model_probabilities_a[model_index]
        proposal_sign = 1.0 if probability_a >= 0.5 else -1.0
        utility_difference = (
            state.actual_net_gain + HYBRID_COST
        ) / state.proposal_sign
        actual_net_gain = proposal_sign * utility_difference - HYBRID_COST
        labels.append(int(actual_net_gain > 0.0))
        probabilities.append(
            probability_a if proposal_sign > 0.0 else 1.0 - probability_a
        )
    return binary_metrics(labels, probabilities, threshold=0.5)


def _component_ablation_metrics(
    feedback: list[HybridProductState],
    test: list[HybridProductState],
    mode: str,
) -> dict[str, float]:
    feedback_observations = [_observation_for_mode(state, mode) for state in feedback]
    test_observations = [_observation_for_mode(state, mode) for state in test]
    calibrator = fit_continuous_gain_calibrator(
        feedback_observations,
        use_source_variance=mode != "mean_only",
    )
    predictions = calibrator.predict(test_observations)
    return binary_metrics(
        [int(state.actual_net_gain > 0.0) for state in test],
        [prediction.action_worthiness for prediction in predictions],
        threshold=0.5,
    )


def _observation_for_mode(
    state: HybridProductState, mode: str
) -> ContinuousGainObservation:
    if mode not in {"mean_only", "internal_only", "internal_external", "full"}:
        raise ValueError(f"unknown component ablation mode: {mode}")
    return ContinuousGainObservation(
        record_id=state.record_id,
        benchmark="lfp_real_hybrid_product_gain",
        mean_gain=state.predicted_mean_net_gain,
        internal_variance=(
            state.internal_variance
            if mode in {"internal_only", "internal_external", "full"}
            else 0.0
        ),
        external_variance=(
            state.external_variance
            if mode in {"internal_external", "full"}
            else 0.0
        ),
        interaction_variance=state.interaction_variance if mode == "full" else 0.0,
        gain=state.actual_net_gain,
    )


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0
