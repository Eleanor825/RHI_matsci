from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class GainHarnessContract:
    name: str
    gain_target: str
    allocation_scope: str
    score: str
    source_contract: tuple[str, ...]
    budget_fraction: float = 0.10
    minimum_score: float | None = None

    def __post_init__(self) -> None:
        if self.gain_target not in {"budget_conditioned", "strict_realized"}:
            raise ValueError("unknown gain target")
        if self.allocation_scope not in {"per_task_quota", "global_cap"}:
            raise ValueError("unknown allocation scope")
        if self.score not in {
            "action_worthiness",
            "expected_gain",
            "conformal_lower_gain",
        }:
            raise ValueError("unknown gain score")
        if not self.source_contract:
            raise ValueError("source contract cannot be empty")
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError("budget fraction must lie in (0, 1]")

    def to_json(self):
        return asdict(self)


@dataclass(frozen=True)
class GainFailureAudit:
    selected_mean_gain: float
    selective_risk: float
    risk_upper_bound: float
    alpha: float
    conformal_coverage: float
    conformal_target: float
    forced_negative_task_count: int
    external_variance_error_correlation: float

    def to_json(self):
        return asdict(self)


@dataclass(frozen=True)
class GainContractProposal:
    predecessor: GainHarnessContract
    candidate: GainHarnessContract
    mutations: tuple[str, ...]
    triggers: tuple[str, ...]

    def to_json(self):
        return {
            "predecessor": self.predecessor.to_json(),
            "candidate": self.candidate.to_json(),
            "mutations": list(self.mutations),
            "triggers": list(self.triggers),
        }


@dataclass(frozen=True)
class GainAcceptanceEvidence:
    selected_count: int
    selected_mean_gain: float
    selected_total_gain: float
    selective_risk: float
    risk_upper_bound: float
    alpha: float

    @property
    def certified(self) -> bool:
        return (
            self.selected_count > 0
            and self.selected_mean_gain > 0.0
            and self.selected_total_gain > 0.0
            and self.risk_upper_bound <= self.alpha
        )

    def to_json(self):
        return {**asdict(self), "certified": self.certified}


def propose_gain_contract(
    previous: GainHarnessContract,
    audit: GainFailureAudit,
    *,
    iteration: int,
) -> GainContractProposal:
    candidate = replace(previous, name=f"H{iteration}_gain_contract")
    mutations = []
    triggers = []
    if audit.selected_mean_gain <= 0.0 and previous.gain_target == "budget_conditioned":
        candidate = replace(candidate, gain_target="strict_realized")
        mutations.append("separate realized gain from resource shadow price")
        triggers.append("selected actions had non-positive realized scientific gain")
    if (
        audit.forced_negative_task_count > 0
        and previous.allocation_scope == "per_task_quota"
    ):
        candidate = replace(candidate, allocation_scope="global_cap")
        mutations.append("replace per-task quota with a global at-most budget")
        triggers.append("quota forced execution in tasks without positive candidates")
    if audit.risk_upper_bound > audit.alpha:
        candidate = replace(
            candidate,
            score="conformal_lower_gain",
            minimum_score=0.0,
        )
        mutations.append("rank by positive conformal lower gain")
        triggers.append("selective-risk upper bound exceeded the target")
    if (
        audit.external_variance_error_correlation > 0.2
        and "verification" not in candidate.source_contract
    ):
        candidate = replace(
            candidate,
            source_contract=(*candidate.source_contract, "verification"),
        )
        mutations.append("add verification evidence replicas")
        triggers.append("external variance predicted gain error")
    if audit.conformal_coverage < audit.conformal_target:
        triggers.append("conformal coverage requires recalibration on fresh feedback")
    return GainContractProposal(
        predecessor=previous,
        candidate=candidate,
        mutations=tuple(mutations),
        triggers=tuple(triggers),
    )


def accept_gain_contract(
    proposal: GainContractProposal,
    predecessor: GainAcceptanceEvidence,
    candidate: GainAcceptanceEvidence,
    *,
    minimum_gain_improvement: float = 0.0,
) -> dict[str, object]:
    accepted = candidate.certified and (
        not predecessor.certified
        or candidate.selected_total_gain
        > predecessor.selected_total_gain + minimum_gain_improvement
    )
    return {
        "accepted": accepted,
        "deployed_contract": (
            proposal.candidate.to_json()
            if accepted
            else proposal.predecessor.to_json()
        ),
        "proposal": proposal.to_json(),
        "predecessor_evidence": predecessor.to_json(),
        "candidate_evidence": candidate.to_json(),
        "acceptance_rule": (
            "candidate must have positive gain, non-empty selection, exact risk "
            "certificate, and improve total gain over any certified predecessor"
        ),
    }
