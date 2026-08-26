from harness_matsci.gain_contract_evolution import (
    GainAcceptanceEvidence,
    GainFailureAudit,
    GainHarnessContract,
    accept_gain_contract,
    propose_gain_contract,
)


def test_failure_feedback_mutates_h2_into_strict_safe_contract():
    previous = GainHarnessContract(
        name="H2",
        gain_target="budget_conditioned",
        allocation_scope="per_task_quota",
        score="action_worthiness",
        source_contract=("screening",),
    )
    proposal = propose_gain_contract(
        previous,
        GainFailureAudit(
            selected_mean_gain=-0.15,
            selective_risk=0.39,
            risk_upper_bound=0.58,
            alpha=0.10,
            conformal_coverage=0.84,
            conformal_target=0.90,
            forced_negative_task_count=1,
            external_variance_error_correlation=0.57,
        ),
        iteration=3,
    )

    assert proposal.candidate.gain_target == "strict_realized"
    assert proposal.candidate.allocation_scope == "global_cap"
    assert proposal.candidate.score == "conformal_lower_gain"
    assert proposal.candidate.source_contract == ("screening", "verification")
    assert proposal.candidate.minimum_score == 0.0


def test_uncertified_candidate_cannot_replace_predecessor():
    previous = GainHarnessContract(
        name="H2",
        gain_target="budget_conditioned",
        allocation_scope="per_task_quota",
        score="action_worthiness",
        source_contract=("screening",),
    )
    proposal = propose_gain_contract(
        previous,
        GainFailureAudit(-0.1, 0.4, 0.6, 0.1, 0.8, 0.9, 1, 0.3),
        iteration=3,
    )
    result = accept_gain_contract(
        proposal,
        GainAcceptanceEvidence(20, -0.1, -2.0, 0.4, 0.6, 0.1),
        GainAcceptanceEvidence(22, 0.8, 17.6, 0.0, 0.13, 0.1),
    )

    assert result["accepted"] is False
    assert result["deployed_contract"]["name"] == "H2"


def test_certified_positive_candidate_is_accepted():
    previous = GainHarnessContract(
        name="H2",
        gain_target="budget_conditioned",
        allocation_scope="per_task_quota",
        score="action_worthiness",
        source_contract=("screening",),
    )
    proposal = propose_gain_contract(
        previous,
        GainFailureAudit(-0.1, 0.4, 0.6, 0.1, 0.8, 0.9, 1, 0.3),
        iteration=3,
    )
    result = accept_gain_contract(
        proposal,
        GainAcceptanceEvidence(20, -0.1, -2.0, 0.4, 0.6, 0.1),
        GainAcceptanceEvidence(40, 0.8, 32.0, 0.0, 0.08, 0.1),
    )

    assert result["accepted"] is True
    assert result["deployed_contract"]["name"] == "H3_gain_contract"
