import json

import pytest

from harness_matsci.llm_worthiness import PROMPT_VERSION
from harness_matsci.sequential_safe_evidence import (
    build_factorized_gain_states,
    build_sequential_evidence_states,
    load_worthiness_prediction_caches,
    load_worthiness_probability_caches,
)


def _trajectory():
    return {
        "record_id": "r1",
        "benchmark": "task",
        "hidden_outcome_for_evaluation_only": {
            "base_gain": 0.7,
            "budget_conditioned_gain": 0.2,
            "reservation_value": 0.5,
            "success_label": 1,
            "normalized_utility": 0.8,
        },
        "stages": [
            {
                "stage": "base",
                "tool_cost": 0.0,
                "visible_observation": {
                    "record": {
                        "metadata": {"group_id": "g1"},
                        "features": {"cost": 0.2, "reversibility": 0.8},
                    }
                },
            },
            {
                "stage": "screening",
                "tool_cost": 0.025,
                "visible_observation": {
                    "record": {
                        "features": {
                            "tool_estimated_margin": 0.2,
                            "tool_estimated_primary": 0.7,
                            "tool_estimated_secondary": 0.6,
                            "tool_positive_probability": 0.75,
                            "tool_uncertainty": 0.1,
                            "tool_fidelity": 0.5,
                        }
                    }
                },
            },
            {
                "stage": "verification",
                "tool_cost": 0.075,
                "visible_observation": {
                    "record": {
                        "features": {
                            "tool_estimated_margin": 0.4,
                            "tool_estimated_primary": 0.8,
                            "tool_estimated_secondary": 0.7,
                            "tool_positive_probability": 0.9,
                            "tool_uncertainty": 0.05,
                            "tool_fidelity": 0.9,
                        }
                    }
                },
            },
        ],
    }


def test_builds_crossed_tool_distribution():
    state = build_sequential_evidence_states(
        [_trajectory()],
        {"r1": (0.6, 0.7, 0.8)},
    )[0]

    assert state.candidate_group_ids == ("g1",)
    assert state.proposal_aligned_source_matrix == (
        (0.1, 0.35000000000000003),
        (0.2, 0.4),
        (0.30000000000000004, 0.45),
    )


def test_loads_complete_worthiness_grid(tmp_path):
    paths = {}
    for model, probability in (("m1", 0.6), ("m2", 0.7)):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "predictions": {
                        "r1::sequential-v2::base": {
                            "p_positive_gain": probability,
                            "p_success": probability + 0.05,
                            "expected_normalized_utility": probability + 0.1,
                        }
                    },
                }
            )
        )
        paths[model] = path

    loaded = load_worthiness_probability_caches(
        [_trajectory()], model_cache_paths=paths
    )

    assert loaded == {"r1": (0.6, 0.7)}

    full = load_worthiness_prediction_caches(
        [_trajectory()], model_cache_paths=paths
    )
    states = build_factorized_gain_states([_trajectory()], full)
    assert states[0].model_success_probabilities == (0.65, 0.75)
    assert states[0].model_utility_means == (0.7, 0.7999999999999999)
    assert len(states[0].source_success_probabilities) == 6
    assert states[0].source_names == (
        "screening::q-1.0",
        "screening::q+0.0",
        "screening::q+1.0",
        "verification::q-1.0",
        "verification::q+0.0",
        "verification::q+1.0",
    )
    assert states[0].failure_harm == pytest.approx(0.2)


def test_strict_realized_gain_does_not_subtract_budget_reservation(tmp_path):
    paths = {}
    for model in ("m1", "m2"):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "predictions": {
                        "r1::sequential-v2::base": {
                            "p_positive_gain": 0.7,
                            "p_success": 0.8,
                            "expected_normalized_utility": 0.9,
                        }
                    },
                }
            )
        )
        paths[model] = path

    predictions = load_worthiness_prediction_caches(
        [_trajectory()], model_cache_paths=paths
    )
    state = build_factorized_gain_states(
        [_trajectory()], predictions, gain_target="strict_realized"
    )[0]

    assert state.reservation_value == 0.0
    assert state.actual_net_gain == pytest.approx(0.7)


def test_factorized_state_can_charge_acquired_source_costs(tmp_path):
    paths = {}
    for model in ("m1", "m2"):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "predictions": {
                        "r1::sequential-v2::base": {
                            "p_positive_gain": 0.7,
                            "p_success": 0.8,
                            "expected_normalized_utility": 0.9,
                        }
                    },
                }
            )
        )
        paths[model] = path
    predictions = load_worthiness_prediction_caches(
        [_trajectory()], model_cache_paths=paths
    )

    free = build_factorized_gain_states(
        [_trajectory()],
        predictions,
        gain_target="strict_realized",
    )[0]
    charged = build_factorized_gain_states(
        [_trajectory()],
        predictions,
        gain_target="strict_realized",
        include_source_cost=True,
    )[0]

    assert charged.normalized_cost == pytest.approx(free.normalized_cost + 0.1)
