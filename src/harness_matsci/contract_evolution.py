from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContractCandidate:
    name: str
    contract_family: str
    acquisition_head: str
    fold_certified: tuple[bool, ...]
    fold_deployed_gains: tuple[float, ...]

    @property
    def certification_rate(self):
        return sum(self.fold_certified) / len(self.fold_certified)

    @property
    def mean_deployed_gain(self):
        return sum(self.fold_deployed_gains) / len(self.fold_deployed_gains)

    @property
    def worst_deployed_gain(self):
        return min(self.fold_deployed_gains)

    def to_json(self):
        return {
            **asdict(self),
            "certification_rate": self.certification_rate,
            "mean_deployed_gain": self.mean_deployed_gain,
            "worst_deployed_gain": self.worst_deployed_gain,
        }


@dataclass(frozen=True)
class ContractEvolutionResult:
    selected: ContractCandidate
    candidates: tuple[ContractCandidate, ...]
    trace: tuple[dict, ...]
    required_certification_rate: float

    def to_json(self):
        return {
            "selected": self.selected.to_json(),
            "candidates": [row.to_json() for row in self.candidates],
            "trace": list(self.trace),
            "required_certification_rate": self.required_certification_rate,
        }


def evolve_contract(
    candidates: list[ContractCandidate],
    *,
    required_certification_rate: float = 1.0,
) -> ContractEvolutionResult:
    if not candidates:
        raise ValueError("contract evolution requires candidates")
    if not 0.0 < required_certification_rate <= 1.0:
        raise ValueError("required certification rate must lie in (0, 1]")
    fold_counts = {len(row.fold_certified) for row in candidates}
    if len(fold_counts) != 1 or any(
        len(row.fold_certified) != len(row.fold_deployed_gains) for row in candidates
    ):
        raise ValueError("all contract candidates must use aligned folds")
    incumbent = None
    trace = []
    for cycle, candidate in enumerate(candidates):
        feasible = candidate.certification_rate >= required_certification_rate
        accepted = feasible and (
            incumbent is None
            or (
                candidate.mean_deployed_gain,
                candidate.worst_deployed_gain,
                -len(candidate.name),
            )
            > (
                incumbent.mean_deployed_gain,
                incumbent.worst_deployed_gain,
                -len(incumbent.name),
            )
        )
        if accepted:
            incumbent = candidate
        trace.append(
            {
                "cycle": cycle,
                "candidate": candidate.name,
                "feasible": feasible,
                "accepted": accepted,
                "certification_rate": candidate.certification_rate,
                "mean_deployed_gain": candidate.mean_deployed_gain,
                "incumbent_after_cycle": incumbent.name if incumbent else None,
            }
        )
    if incumbent is None:
        raise RuntimeError("no evolved contract satisfies the certification gate")
    return ContractEvolutionResult(
        selected=incumbent,
        candidates=tuple(candidates),
        trace=tuple(trace),
        required_certification_rate=required_certification_rate,
    )
