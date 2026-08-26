import json

import pytest

from harness_matsci.factorized_gain_harness import FactorizedGainPrediction
from harness_matsci.matbot_strict_gain_runtime import (
    append_runtime_event,
    strict_gain_post_outcome_event,
    strict_gain_pre_action_event,
)
from harness_matsci.matbot_trajectory import assemble_matbot_trajectories


def _prediction():
    return FactorizedGainPrediction(
        record_id="r1",
        benchmark="lfp",
        action_worthiness=0.8,
        expected_net_gain=0.3,
        success_probability=0.75,
        conditional_utility_mean=0.6,
        probability_internal_variance=0.01,
        probability_external_variance=0.02,
        probability_interaction_variance=0.005,
        gain_internal_variance=0.015,
        gain_external_variance=0.025,
        gain_interaction_variance=0.01,
        aleatoric_gain_variance=0.04,
        total_gain_variance=0.09,
    )


def test_builds_numeric_pre_action_and_delayed_strict_outcome(tmp_path):
    pre = strict_gain_pre_action_event(
        event_id="pre-0",
        run_id="run-1",
        trajectory_id="traj-1",
        step_id=0,
        timestamp="2026-08-23T00:00:00Z",
        system="lfp",
        stage="screening",
        action_id="candidate-a",
        action_type="execute_candidate",
        visible_context="visible candidate context",
        evidence=["simulator estimate"],
        internal_signals={"replica_disagreement": 0.1},
        external_signals={"source_disagreement": 0.2},
        normalized_cost=0.2,
        reversibility=0.7,
        prediction=_prediction(),
        conformal_lower_gain=0.05,
    )
    post = strict_gain_post_outcome_event(
        event_id="post-0",
        parent_event_id="pre-0",
        run_id="run-1",
        trajectory_id="traj-1",
        step_id=0,
        timestamp="2026-08-23T01:00:00Z",
        observed_success=1,
        normalized_utility=0.8,
        realized_cost=0.2,
        failure_harm=0.0,
        terminal=True,
    )

    trajectory = assemble_matbot_trajectories([pre, post])[0]
    assert "selected_route" not in pre["uncertainty_output"]
    assert post["observed_outcome"]["strict_realized_net_gain"] == pytest.approx(0.77)
    assert trajectory.complete is True

    path = tmp_path / "runtime.jsonl"
    append_runtime_event(path, pre)
    append_runtime_event(path, post)
    assert len(path.read_text().splitlines()) == 2
    assert json.loads(path.read_text().splitlines()[0])["hidden_outcome_exposed"] is False


def test_failure_gain_includes_irreversible_harm():
    post = strict_gain_post_outcome_event(
        event_id="post-0",
        parent_event_id="pre-0",
        run_id="run-1",
        trajectory_id="traj-1",
        step_id=0,
        timestamp="2026-08-23T01:00:00Z",
        observed_success=0,
        normalized_utility=0.9,
        realized_cost=0.4,
        failure_harm=0.8,
        terminal=True,
    )

    assert post["observed_outcome"]["strict_realized_net_gain"] == pytest.approx(-0.26)
