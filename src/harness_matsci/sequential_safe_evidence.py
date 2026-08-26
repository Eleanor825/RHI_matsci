from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from .factorized_gain_harness import FactorizedGainState
from .llm_worthiness import PROMPT_VERSION
from .safe_evidence_residual import EvidenceResidualState


def load_worthiness_probability_caches(
    trajectories: Sequence[dict[str, object]],
    *,
    model_cache_paths: dict[str, str | Path],
    view_id: str = "sequential-v2::base",
) -> dict[str, tuple[float, ...]]:
    full = load_worthiness_prediction_caches(
        trajectories,
        model_cache_paths=model_cache_paths,
        view_id=view_id,
    )
    return {
        record_id: tuple(float(cell["p_positive_gain"]) for cell in cells)
        for record_id, cells in full.items()
    }


def load_worthiness_prediction_caches(
    trajectories: Sequence[dict[str, object]],
    *,
    model_cache_paths: dict[str, str | Path],
    view_id: str = "sequential-v2::base",
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, tuple[dict[str, object], ...]]:
    predictions_by_model = {}
    for model, path in model_cache_paths.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != model:
            raise ValueError(f"cache model mismatch for {model}")
        if payload.get("prompt_version") != prompt_version:
            raise ValueError(f"cache prompt version mismatch for {model}")
        predictions_by_model[model] = payload.get("predictions", {})
    output: dict[str, tuple[dict[str, object], ...]] = {}
    for trajectory in trajectories:
        record_id = str(trajectory["record_id"])
        key = f"{record_id}::{view_id}"
        values: list[dict[str, object]] = []
        for model in model_cache_paths:
            prediction = predictions_by_model[model].get(key)
            if prediction is None:
                raise ValueError(f"missing worthiness prediction: {model}, {record_id}")
            values.append(dict(prediction))
        output[record_id] = tuple(values)
    return output


def build_sequential_evidence_states(
    trajectories: Sequence[dict[str, object]],
    llm_probabilities: dict[str, tuple[float, ...]],
    *,
    source_stages: tuple[str, ...] = ("screening", "verification"),
) -> list[EvidenceResidualState]:
    if not source_stages:
        raise ValueError("at least one scientific source stage is required")
    output = []
    for trajectory in trajectories:
        record_id = str(trajectory["record_id"])
        stages = {str(row["stage"]): row for row in trajectory["stages"]}
        means = []
        uncertainties = []
        fidelities = []
        for stage in source_stages:
            features = stages[stage]["visible_observation"]["record"]["features"]
            means.append(float(features["tool_estimated_margin"]))
            uncertainties.append(max(1e-6, float(features["tool_uncertainty"])))
            fidelities.append(max(1e-6, float(features["tool_fidelity"])))
        source_matrix = tuple(
            tuple(
                mean + quantile * uncertainty
                for mean, uncertainty in zip(means, uncertainties)
            )
            for quantile in (-1.0, 0.0, 1.0)
        )
        base_metadata = stages["base"]["visible_observation"]["record"].get(
            "metadata", {}
        )
        target = trajectory["hidden_outcome_for_evaluation_only"]
        output.append(
            EvidenceResidualState(
                record_id=record_id,
                benchmark=str(trajectory["benchmark"]),
                objective=str(trajectory["benchmark"]),
                candidate_group_ids=(
                    str(base_metadata.get("group_id", record_id)),
                ),
                model_success_probabilities=tuple(llm_probabilities[record_id]),
                proposal_aligned_source_matrix=source_matrix,
                source_names=source_stages,
                source_weights=tuple(fidelities),
                actual_net_gain=float(target["budget_conditioned_gain"]),
            )
        )
    return output


def build_factorized_gain_states(
    trajectories: Sequence[dict[str, object]],
    llm_predictions: dict[str, tuple[dict[str, object], ...]],
    *,
    source_stages: tuple[str, ...] = ("screening", "verification"),
    gain_target: str = "budget_conditioned",
    include_source_cost: bool = False,
    source_quantiles: tuple[tuple[float, float], ...] = (
        (-1.0, 0.25),
        (0.0, 0.50),
        (1.0, 0.25),
    ),
) -> list[FactorizedGainState]:
    if not source_stages:
        raise ValueError("at least one scientific source stage is required")
    if gain_target not in {"budget_conditioned", "strict_realized"}:
        raise ValueError(f"unknown gain target: {gain_target}")
    output = []
    for trajectory in trajectories:
        record_id = str(trajectory["record_id"])
        stages = {str(row["stage"]): row for row in trajectory["stages"]}
        source_success = []
        source_features = []
        source_weights = []
        for stage in source_stages:
            features = stages[stage]["visible_observation"]["record"]["features"]
            probability = float(features["tool_positive_probability"])
            uncertainty = max(1e-6, float(features["tool_uncertainty"]))
            fidelity = max(1e-6, float(features["tool_fidelity"]))
            for quantile, quantile_weight in source_quantiles:
                source_success.append(
                    1.0
                    / (
                        1.0
                        + math.exp(
                            -(
                                math.log(
                                    min(1.0 - 1e-9, max(1e-9, probability))
                                    / (
                                        1.0
                                        - min(
                                            1.0 - 1e-9,
                                            max(1e-9, probability),
                                        )
                                    )
                                )
                                + quantile
                            )
                        )
                    )
                )
                source_features.append(
                    (
                        float(features["tool_estimated_primary"])
                        + quantile * uncertainty,
                        float(features["tool_estimated_secondary"])
                        + quantile * uncertainty,
                        float(features["tool_estimated_margin"])
                        + quantile * uncertainty,
                    )
                )
                source_weights.append(fidelity * quantile_weight)
        base = stages["base"]["visible_observation"]["record"]
        base_features = base["features"]
        cost = min(1.0, max(0.0, float(base_features.get("cost", 0.5))))
        if include_source_cost:
            cost += sum(float(stages[stage]["tool_cost"]) for stage in source_stages)
        reversibility = min(
            1.0, max(0.0, float(base_features.get("reversibility", 0.5)))
        )
        failure_harm = 0.5 * cost + 0.5 * (1.0 - reversibility)
        metadata = base.get("metadata", {})
        target = trajectory["hidden_outcome_for_evaluation_only"]
        cells = llm_predictions[record_id]
        if gain_target == "strict_realized":
            reservation_value = 0.0
            actual_net_gain = float(target["base_gain"])
        else:
            reservation_value = float(target["reservation_value"])
            actual_net_gain = float(target["budget_conditioned_gain"])
        output.append(
            FactorizedGainState(
                record_id=record_id,
                benchmark=str(trajectory["benchmark"]),
                objective=str(trajectory["benchmark"]),
                group_id=str(metadata.get("group_id", record_id)),
                model_worthiness_probabilities=tuple(
                    float(cell["p_positive_gain"]) for cell in cells
                ),
                model_success_probabilities=tuple(
                    float(cell["p_success"]) for cell in cells
                ),
                model_utility_means=tuple(
                    float(cell["expected_normalized_utility"]) for cell in cells
                ),
                source_names=tuple(
                    f"{stage}::q{quantile:+.1f}"
                    for stage in source_stages
                    for quantile, _ in source_quantiles
                ),
                source_success_probabilities=tuple(source_success),
                source_utility_features=tuple(source_features),
                source_weights=tuple(source_weights),
                normalized_cost=cost,
                failure_harm=failure_harm,
                reservation_value=reservation_value,
                actual_success=int(target["success_label"]),
                actual_normalized_utility=float(target["normalized_utility"]),
                actual_net_gain=actual_net_gain,
            )
        )
    return output
