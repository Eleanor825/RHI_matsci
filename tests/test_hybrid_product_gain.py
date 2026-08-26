import json

import pytest

from harness_matsci.hybrid_product_gain import (
    HybridProductState,
    evaluate_hybrid_product_gain,
    load_llm_probability_caches,
)
from harness_matsci.llm_action_choice import PROMPT_VERSION
from harness_matsci.schema import ActionRecord


def _state(index: int, gain: float) -> HybridProductState:
    probability = 0.8 if gain > 0.0 else 0.55
    return HybridProductState(
        record_id=f"r{index}",
        benchmark="task",
        objective="objective",
        proposal_sign=1.0,
        model_probabilities_a=(probability - 0.05, probability, probability + 0.05),
        success_probabilities=(probability - 0.05, probability, probability + 0.05),
        mean_success_probability=probability,
        mean_value_magnitude=0.3,
        predicted_mean_net_gain=(2.0 * probability - 1.0) * 0.3 - 0.01,
        internal_variance=0.001,
        external_variance=0.002,
        interaction_variance=0.0001,
        actual_net_gain=gain,
    )


def test_hybrid_evaluator_returns_calibration_and_certificate():
    feedback = [_state(index, 0.2 if index % 3 else -0.2) for index in range(18)]
    acceptance = [_state(100 + index, 0.2 if index % 4 else -0.2) for index in range(40)]
    test = [_state(200 + index, 0.2 if index % 3 else -0.2) for index in range(20)]

    result = evaluate_hybrid_product_gain(
        feedback=feedback,
        acceptance=acceptance,
        test=test,
    )

    assert result["hybrid_metrics"]["n"] == pytest.approx(20.0)
    assert result["direct_llm_ensemble_metrics"]["n"] == pytest.approx(20.0)
    assert len(result["direct_llm_model_metrics"]) == 3
    assert set(result["component_ablation_metrics"]) == {
        "mean_only",
        "internal_only",
        "internal_external",
        "full",
    }
    assert "risk_upper_bound" in result["certificate"]
    assert 0.0 <= result["test_conformal_coverage"] <= 1.0


def test_load_llm_probability_caches_requires_complete_frozen_grid(tmp_path):
    record = ActionRecord(
        record_id="r1",
        benchmark="task",
        split="test",
        visible_context="state",
        candidate_action="action",
        action_type="choose_candidate",
        evidence=["evidence"],
        features={},
        label=1,
        utility=1.0,
        metadata={},
    )
    cache_paths = {}
    for model, probability in (("m1", 0.7), ("m2", 0.8)):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "predictions": {
                        f"{PROMPT_VERSION}::r1::replica-0": {
                            "p_a_better": probability
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        cache_paths[model] = path

    loaded = load_llm_probability_caches(
        [record], model_cache_paths=cache_paths
    )

    assert loaded == {"r1": (0.7, 0.8)}
