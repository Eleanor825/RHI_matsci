from harness_matsci.learn_then_test_gain import (
    FixedThresholdGainPolicy,
    evaluate_fixed_threshold_gain_policy,
    fit_learn_then_test_gain_policy,
)


def test_fixed_threshold_policy_uses_outcome_independent_thinning():
    policy = FixedThresholdGainPolicy(0.2, 0.2, 7)
    scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    record_ids = [f"r{index}" for index in range(5)]

    first = policy.select(scores, record_ids)
    second = policy.select(scores, record_ids)

    assert first == second
    assert len(first) == 1
    assert all(scores[index] >= 0.2 for index in first)


def test_ltt_returns_only_simultaneously_certified_positive_gain_policy():
    scores = [0.9 - index / 100 for index in range(40)]
    gains = [0.8] * 35 + [-0.3] * 5
    record_ids = [f"r{index}" for index in range(40)]

    policy, evidence = fit_learn_then_test_gain_policy(
        candidate_thresholds=[0.55, 0.7, 0.85],
        feedback_scores=scores,
        feedback_gains=gains,
        feedback_record_ids=record_ids,
        alpha=0.2,
        delta=0.05,
        budget_fraction=1.0,
        thinning_seed=11,
    )

    assert policy is not None
    selected = next(row for row in evidence if row.threshold == policy.threshold)
    assert selected.certified
    assert selected.simultaneous_risk_upper_bound <= 0.2


def test_frozen_threshold_evaluation_reports_exact_assumption_boundary():
    policy = FixedThresholdGainPolicy(0.5, 1.0, 3)
    result = evaluate_fixed_threshold_gain_policy(
        policy,
        scores=[0.9] * 30 + [0.1],
        gains=[0.5] * 30 + [-0.5],
        record_ids=[f"r{index}" for index in range(31)],
        alpha=0.1,
        delta=0.05,
    )

    assert result["selected_count"] == 30
    assert result["errors"] == 0
    assert result["risk_upper_bound"] <= 0.1
    assert "i.i.d." in result["guarantee"]
