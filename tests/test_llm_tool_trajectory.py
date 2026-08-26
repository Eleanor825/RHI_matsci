from __future__ import annotations

from harness_matsci.llm_tool_trajectory import (
    ToolPolicyDecision,
    run_model_tool_trajectory,
)


class FakePolicy:
    model = "fake-model"

    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, **kwargs):
        action = next(self.actions)
        assert action in kwargs["allowed_actions"]
        assert "hidden_outcome" not in kwargs["user_prompt"]
        return ToolPolicyDecision(action, 0.7, "test")


def _source():
    def stage(name, cost, evidence):
        return {
            "stage": name,
            "tool_cost": cost,
            "visible_observation": {
                "record": {
                    "candidate_action": "Execute candidate A.",
                    "visible_context": "Visible context only.",
                    "evidence": evidence,
                },
                "provenance": {"hidden_outcome_exposed": False},
            },
        }

    return {
        "record_id": "r1",
        "benchmark": "task",
        "split": "test",
        "stages": [
            stage("base", 0.0, ["base evidence"]),
            stage("screening", 0.1, ["screen evidence"]),
            stage("verification", 0.2, ["verify evidence"]),
        ],
        "hidden_outcome_for_evaluation_only": {
            "base_gain": 0.8,
            "success_label": 1,
            "normalized_utility": 0.9,
        },
    }


def test_model_tool_trajectory_links_tool_cost_and_hidden_outcome_after_execution():
    result = run_model_tool_trajectory(
        _source(),
        FakePolicy(["acquire_screening", "execute_candidate"]),
    )

    assert result["terminal_action"] == "execute_candidate"
    assert result["evidence_cost"] == 0.1
    assert result["hidden_outcome_for_evaluation_only"]["trajectory_strict_net_gain"] == 0.785
    assert all(not step["hidden_outcome_exposed"] for step in result["steps"])
    assert result["credit_assignment"]["causal_action_value_claimed"] is False


def test_model_tool_trajectory_charges_evidence_before_abstention():
    result = run_model_tool_trajectory(
        _source(),
        FakePolicy(["acquire_verification", "abstain"]),
    )

    assert result["terminal_action"] == "abstain"
    assert result["executed_candidate"] is False
    assert result["hidden_outcome_for_evaluation_only"]["trajectory_strict_net_gain"] == -0.03
