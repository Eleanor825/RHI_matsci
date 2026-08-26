from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .schema import ActionRecord


COMPONENTS = (
    "EC",
    "DMC",
    "EMC",
    "EA",
    "MA",
    "PC",
    "ACN",
    "戊腈",
    "LiPF6",
    "LiFSI",
    "LiFSO3",
    "DTD",
    "FEC",
    "VC",
    "TMSP",
    "ES",
    "MMDS",
    "LiPO2F2",
    "LiODFP",
    "LiODFB",
    "EC2DTD",
    "BiDTD",
)
SALT_COMPONENTS = frozenset({"LiPF6", "LiFSI", "LiFSO3"})
ADDITIVE_COMPONENTS = frozenset(
    {"DTD", "FEC", "VC", "TMSP", "ES", "MMDS", "LiPO2F2", "LiODFP", "LiODFB", "EC2DTD", "BiDTD"}
)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    source_field: str
    higher_is_better: bool
    description: str


OBJECTIVES = (
    ObjectiveSpec("conductivity", "电导率", True, "maximize electrolyte conductivity"),
    ObjectiveSpec("formation_gas", "化成产气量\n(g)", False, "minimize formation gas"),
    ObjectiveSpec("first_efficiency", "电芯下线首效", True, "maximize first-cycle efficiency"),
    ObjectiveSpec(
        "specific_capacity",
        "电芯下线放电材料比容量",
        True,
        "maximize first-discharge specific capacity",
    ),
    ObjectiveSpec(
        "internal_resistance",
        "电芯下线内阻",
        False,
        "minimize end-of-line internal resistance",
    ),
    ObjectiveSpec("dcr_25c_50soc", "25℃ 50%SOC DCR", False, "minimize 25C 50% SOC DCR"),
    ObjectiveSpec("dcr_minus20c_50soc", "-20℃ 50%SOC DCR", False, "minimize -20C 50% SOC DCR"),
)


@dataclass(frozen=True)
class LfpFormulationExperiment:
    experiment_id: str
    family: str
    composition: dict[str, float]
    outcomes: dict[str, float]
    source_rows: tuple[int, int]


def load_lfp_formulation_experiments(path: str | Path) -> list[LfpFormulationExperiment]:
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paired: dict[str, dict[str, tuple[int, dict[str, str]]]] = defaultdict(dict)
    for row_index, row in enumerate(rows):
        name = str(row.get("牌号", "")).strip()
        match = re.fullmatch(r"(.+?)([AB])", name)
        if match:
            paired[match.group(1)][match.group(2)] = (row_index, row)
    experiments = []
    for experiment_id, pair in sorted(paired.items()):
        if set(pair) != {"A", "B"}:
            continue
        a_index, a_row = pair["A"]
        b_index, b_row = pair["B"]
        a_weight = _number(a_row.get("注液比例"))
        b_weight = _number(b_row.get("注液比例"))
        if a_weight is None or b_weight is None or a_weight + b_weight <= 0.0:
            continue
        total_weight = a_weight + b_weight
        composition = {
            component: (
                a_weight * (_number(a_row.get(component)) or 0.0)
                + b_weight * (_number(b_row.get(component)) or 0.0)
            )
            / total_weight
            for component in COMPONENTS
        }
        outcomes = {}
        for objective in OBJECTIVES:
            if objective.name == "conductivity":
                left = _number(a_row.get(objective.source_field))
                right = _number(b_row.get(objective.source_field))
                if left is not None and right is not None:
                    outcomes[objective.name] = (
                        a_weight * left + b_weight * right
                    ) / total_weight
                elif left is not None:
                    outcomes[objective.name] = left
                elif right is not None:
                    outcomes[objective.name] = right
                continue
            value = _number(a_row.get(objective.source_field))
            if value is None:
                value = _number(b_row.get(objective.source_field))
            if value is not None:
                outcomes[objective.name] = value
        experiments.append(
            LfpFormulationExperiment(
                experiment_id=experiment_id,
                family=_family(experiment_id),
                composition=composition,
                outcomes=outcomes,
                source_rows=(a_index, b_index),
            )
        )
    return experiments


def build_lfp_candidate_actions(
    experiments: list[LfpFormulationExperiment],
) -> list[ActionRecord]:
    by_group = defaultdict(list)
    for experiment in experiments:
        for objective in OBJECTIVES:
            if objective.name in experiment.outcomes:
                by_group[(experiment.family, objective.name)].append(experiment)
    records = []
    for (family, objective_name), group in sorted(by_group.items()):
        if len(group) < 3:
            continue
        objective = _objective(objective_name)
        values = [experiment.outcomes[objective_name] for experiment in group]
        utilities = {
            experiment.experiment_id: _rank_utility(
                experiment.outcomes[objective_name],
                values,
                higher_is_better=objective.higher_is_better,
            )
            for experiment in group
        }
        threshold = _median(list(utilities.values()))
        for experiment in sorted(group, key=lambda item: item.experiment_id):
            utility = utilities[experiment.experiment_id]
            records.append(
                ActionRecord(
                    record_id=(
                        f"lfp-real::{family}::{objective_name}::{experiment.experiment_id}"
                    ),
                    benchmark=f"lfp_real_{objective_name}",
                    split="real_database_unassigned",
                    visible_context=_visible_context(experiment, objective),
                    candidate_action=(
                        f"Execute formulation {experiment.experiment_id} under the registered "
                        f"{family} protocol to {objective.description}."
                    ),
                    action_type="recommend_experiment",
                    evidence=[
                        "source=real_lfp_formulation_performance_database",
                        f"protocol_family={family}",
                        f"objective={objective_name}",
                        "target_measurement=withheld_until_evaluation",
                    ],
                    features=_features(experiment, objective_name),
                    label=int(utility > threshold),
                    utility=utility,
                    metadata={
                        "source": "real_lfp_formulation_performance_database",
                        "source_batch": f"{family}::{objective_name}",
                        "family": family,
                        "experiment_id": experiment.experiment_id,
                        "objective": objective_name,
                        "objective_direction": (
                            "maximize" if objective.higher_is_better else "minimize"
                        ),
                        "candidate_group_id": experiment.experiment_id,
                        "source_rows": list(experiment.source_rows),
                        "hidden_outcome_for_evaluation_only": {
                            "objective": objective_name,
                            "raw_measurement": experiment.outcomes[objective_name],
                            "normalized_scientific_utility": utility,
                        },
                        "confirmation_status": "real_source_development_external_validation",
                    },
                )
            )
    return records


def build_lfp_pairwise_actions(candidate_records: list[ActionRecord]) -> list[ActionRecord]:
    by_group = defaultdict(list)
    for record in candidate_records:
        by_group[(record.metadata["family"], record.metadata["objective"])].append(record)
    output = []
    for (family, objective), group in sorted(by_group.items()):
        ordered = sorted(group, key=lambda item: item.record_id)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                candidate_a, candidate_b = _stable_orientation(left, right)
                feature_names = sorted(set(candidate_a.features) | set(candidate_b.features))
                deltas = {
                    f"candidate_delta::{name}": (
                        float(candidate_a.features.get(name, 0.0))
                        - float(candidate_b.features.get(name, 0.0))
                    )
                    for name in feature_names
                    if name.startswith("component_fraction::")
                    or name in {"composition_entropy", "salt_fraction", "additive_fraction"}
                }
                output.append(
                    ActionRecord(
                        record_id=(
                            f"lfp-real-pair::{family}::{objective}::"
                            f"{candidate_a.metadata['experiment_id']}::"
                            f"{candidate_b.metadata['experiment_id']}"
                        ),
                        benchmark=f"lfp_real_pair_{objective}",
                        split="real_database_unassigned",
                        visible_context=(
                            f"Real LFP formulation comparison for objective `{objective}`. "
                            "Measured outcomes are withheld.\n\n"
                            f"Candidate A:\n{candidate_a.visible_context}\n\n"
                            f"Candidate B:\n{candidate_b.visible_context}"
                        ),
                        candidate_action="Execute candidate A rather than candidate B.",
                        action_type="choose_candidate",
                        evidence=[
                            "source=real_lfp_formulation_performance_database",
                            f"protocol_family={family}",
                            f"objective={objective}",
                            "candidate_outcomes=withheld_until_evaluation",
                            "candidate_orientation=stable_hash_without_outcomes",
                        ],
                        features={
                            **deltas,
                            "candidate_visible_feature_distance": math.sqrt(
                                sum(value**2 for value in deltas.values())
                            ),
                            "source_reliability": min(
                                candidate_a.features["source_reliability"],
                                candidate_b.features["source_reliability"],
                            ),
                            "evidence_support": 1.0,
                            "protocol_comparability": 1.0,
                            "cost": candidate_a.features["cost"],
                            "reversibility": candidate_a.features["reversibility"],
                        },
                        label=int(candidate_a.utility > candidate_b.utility),
                        utility=candidate_a.utility,
                        metadata={
                            "source": "real_lfp_formulation_performance_database_pairwise",
                            "source_batch": f"{family}::{objective}",
                            "family": family,
                            "objective": objective,
                            "candidate_a_record_id": candidate_a.record_id,
                            "candidate_b_record_id": candidate_b.record_id,
                            "candidate_group_ids": [
                                candidate_a.metadata["experiment_id"],
                                candidate_b.metadata["experiment_id"],
                            ],
                            "orientation_uses_hidden_outcomes": False,
                            "hidden_outcome_for_evaluation_only": {
                                "candidate_a_utility": candidate_a.utility,
                                "candidate_b_utility": candidate_b.utility,
                                "utility_difference": candidate_a.utility - candidate_b.utility,
                                "candidate_a_measurement": candidate_a.metadata[
                                    "hidden_outcome_for_evaluation_only"
                                ]["raw_measurement"],
                                "candidate_b_measurement": candidate_b.metadata[
                                    "hidden_outcome_for_evaluation_only"
                                ]["raw_measurement"],
                            },
                            "confirmation_status": "real_source_development_external_validation",
                        },
                    )
                )
    return output


def _visible_context(
    experiment: LfpFormulationExperiment,
    objective: ObjectiveSpec,
) -> str:
    composition = ", ".join(
        f"{name}={value:.3f}%"
        for name, value in experiment.composition.items()
        if value > 0.0
    )
    return (
        f"Real historical LFP full-cell formulation {experiment.experiment_id}; "
        f"protocol family={experiment.family}; target={objective.description}. "
        f"Weighted formulation: {composition}. The target measurement is hidden."
    )


def _features(
    experiment: LfpFormulationExperiment,
    objective_name: str,
) -> dict[str, float]:
    total = sum(experiment.composition.values())
    probabilities = [
        value / total for value in experiment.composition.values() if value > 0.0
    ]
    entropy = -sum(value * math.log(value) for value in probabilities)
    maximum_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    return {
        **{
            f"component_fraction::{name}": experiment.composition[name] / 100.0
            for name in COMPONENTS
        },
        "composition_entropy": entropy / maximum_entropy if maximum_entropy else 0.0,
        "salt_fraction": sum(
            experiment.composition[name] for name in SALT_COMPONENTS
        )
        / 100.0,
        "additive_fraction": sum(
            experiment.composition[name] for name in ADDITIVE_COMPONENTS
        )
        / 100.0,
        f"objective::{objective_name}": 1.0,
        "source_reliability": 0.90,
        "evidence_support": 1.0,
        "protocol_comparability": 1.0,
        "cost": 0.65,
        "reversibility": 0.35,
    }


def _rank_utility(value: float, values: list[float], *, higher_is_better: bool) -> float:
    better = sum(
        candidate > value if higher_is_better else candidate < value
        for candidate in values
    )
    equal = sum(candidate == value for candidate in values)
    rank = better + 0.5 * equal
    return 1.0 - rank / len(values)


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _family(experiment_id: str) -> str:
    if experiment_id.startswith("AEA"):
        return "AEA"
    if experiment_id.startswith("ACA"):
        return "ACA"
    if experiment_id.startswith("LFP-L"):
        return "LFP-L"
    return "legacy"


def _objective(name: str) -> ObjectiveSpec:
    return next(objective for objective in OBJECTIVES if objective.name == name)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _stable_orientation(
    left: ActionRecord,
    right: ActionRecord,
) -> tuple[ActionRecord, ActionRecord]:
    digest = hashlib.sha256(
        f"{left.record_id}|{right.record_id}".encode("utf-8")
    ).digest()
    return (left, right) if digest[0] % 2 == 0 else (right, left)
