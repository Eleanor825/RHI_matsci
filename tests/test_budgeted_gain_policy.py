import pytest

from harness_matsci.budgeted_gain_policy import (
    BudgetedGainPolicy,
    certify_budgeted_gain_policy,
    evaluate_budgeted_gain_policy,
)


def test_global_budget_is_a_cap_and_never_forces_negative_scores():
    policy = BudgetedGainPolicy(budget_fraction=0.5, minimum_score=0.0)

    selected = policy.select([-0.2, 0.8, 0.4, -0.1])

    assert selected == (1, 2)


def test_policy_can_abstain_below_budget():
    policy = BudgetedGainPolicy(budget_fraction=0.5, minimum_score=0.5)

    report = evaluate_budgeted_gain_policy(
        policy,
        [0.4, 0.7, 0.3, 0.2],
        [-0.5, 0.8, 0.2, 0.1],
    )

    assert report["selected_count"] == 1
    assert report["selected_total_net_gain"] == pytest.approx(0.8)
    assert report["selective_risk"] == 0.0


def test_certificate_is_exact_for_the_pre_fixed_policy():
    policy = BudgetedGainPolicy(budget_fraction=1.0, minimum_score=0.0)
    report = certify_budgeted_gain_policy(
        policy,
        [1.0] * 40,
        [0.2] * 40,
        alpha=0.1,
        delta=0.05,
    )

    assert report["selected_count"] == 40
    assert report["risk_upper_bound"] < 0.1
    assert report["certified"] is True


def test_empty_selection_is_not_a_vacuous_certificate():
    policy = BudgetedGainPolicy(budget_fraction=0.1, minimum_score=1.0)
    report = certify_budgeted_gain_policy(
        policy,
        [0.2, 0.3],
        [0.4, 0.5],
        alpha=0.1,
    )

    assert report["selected_count"] == 0
    assert report["certified"] is False
    assert report["risk_upper_bound"] == 1.0
