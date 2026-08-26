from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from .matbot_trajectory import SCHEMA_VERSION, MatBotTrajectory, assemble_matbot_trajectories
from .schema import ActionRecord


TASK = "matbot_lfp_historical_action"
CHOICE_TASK = "matbot_lfp_historical_choice"
COMPONENTS = (
    "EC",
    "DMC",
    "EMC",
    "EA",
    "MA",
    "PC",
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
    "LiODFP",
    "LiODFB",
)
SALT_COMPONENTS = {"LiPF6", "LiFSI", "LiFSO3"}
ADDITIVE_COMPONENTS = {"DTD", "FEC", "VC", "TMSP", "ES", "MMDS", "LiODFP", "LiODFB"}


def load_matbot_lfp_historical_actions(path: str | Path) -> list[ActionRecord]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("historical LFP source is empty")
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[row["source_batch"]].append(row)
    utilities = {}
    outcomes = {}
    for batch, batch_rows in by_batch.items():
        batch_outcomes = [_outcome(row) for row in batch_rows]
        dcr_values = [outcome["primary_dcr"] for outcome in batch_outcomes]
        cycle_values = [outcome["cycle_600"] for outcome in batch_outcomes if outcome["cycle_600"] is not None]
        for row, outcome in zip(batch_rows, batch_outcomes):
            dcr_score = _rank_score(outcome["primary_dcr"], dcr_values, higher_is_better=False)
            objective_scores = [dcr_score]
            if outcome["cycle_600"] is not None:
                objective_scores.append(
                    _rank_score(outcome["cycle_600"], cycle_values, higher_is_better=True)
                )
            utility = sum(objective_scores) / len(objective_scores)
            utilities[row["pair_id"]] = utility
            outcomes[row["pair_id"]] = {
                **outcome,
                "dcr_rank_utility": dcr_score,
                "normalized_scientific_utility": utility,
            }
    records = []
    for row in rows:
        composition = _composition(row)
        utility = utilities[row["pair_id"]]
        batch_utilities = [utilities[item["pair_id"]] for item in by_batch[row["source_batch"]]]
        threshold = _median(batch_utilities)
        records.append(
            ActionRecord(
                record_id=f"matbot-lfp-history::{row['source_batch']}::{_slug(row['pair_id'])}",
                benchmark=TASK,
                split="development_replay",
                visible_context=_visible_context(row, composition),
                candidate_action=(
                    f"Execute historical LFP formulation pair {row['pair_id']} under its "
                    "registered battery test protocol."
                ),
                action_type="recommend_experiment",
                evidence=[
                    f"source_document={row['source_doc']}",
                    f"source_batch={row['source_batch']}",
                    "performance_outcome=withheld_until_evaluation",
                ],
                features=_visible_features(row, composition),
                label=int(utility > threshold),
                utility=utility,
                metadata={
                    "source": "real_matbot_historical_experiment",
                    "source_batch": row["source_batch"],
                    "pair_id": row["pair_id"],
                    "hidden_outcome_for_evaluation_only": outcomes[row["pair_id"]],
                    "utility_definition": (
                        "mean within-protocol empirical rank of lower primary DCR and, "
                        "when available, higher 45C cycle-600 retention"
                    ),
                    "confirmation_status": "development_only_source_inspected_before_protocol_lock",
                },
            )
        )
    return records


def build_matbot_lfp_historical_trajectories(
    records: list[ActionRecord],
) -> list[MatBotTrajectory]:
    events = []
    for record in records:
        trajectory_id = f"historical-replay::{record.record_id}"
        event_id = f"historical-pre::{record.record_id}"
        external_names = {"source_reliability", "evidence_support", "protocol_comparability"}
        cost_names = {"cost", "reversibility"}
        events.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_type": "pre_action",
                "event_id": event_id,
                "run_id": str(record.metadata["source_batch"]),
                "trajectory_id": trajectory_id,
                "step_id": 0,
                "timestamp": "",
                "capture_mode": "offline_replay",
                "system": "lfp",
                "stage": "historical_replay",
                "action_id": record.record_id,
                "action_type": record.action_type,
                "visible_context": record.visible_context,
                "evidence": record.evidence,
                "internal_signals": {},
                "external_signals": {
                    name: value for name, value in record.features.items() if name in external_names
                },
                "cost_signals": {
                    name: value for name, value in record.features.items() if name in cost_names
                },
                "uncertainty_output": {},
                "action_text": record.candidate_action,
                "rationale": "historically executed formulation; outcome withheld in replay",
                "hidden_outcome_exposed": False,
            }
        )
        events.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_type": "post_outcome",
                "event_id": f"historical-post::{record.record_id}",
                "parent_event_id": event_id,
                "run_id": str(record.metadata["source_batch"]),
                "trajectory_id": trajectory_id,
                "step_id": 0,
                "timestamp": "",
                "capture_mode": "offline_replay",
                "outcome_status": "measured",
                "observed_outcome": {
                    "action_worthy_label": record.label,
                    "scientific_utility": record.utility,
                    **record.metadata["hidden_outcome_for_evaluation_only"],
                },
                "realized_cost": record.features["cost"],
                "terminal": True,
            }
        )
    return assemble_matbot_trajectories(events)


def build_matbot_lfp_historical_choices(
    records: list[ActionRecord],
) -> list[ActionRecord]:
    by_batch = defaultdict(list)
    for record in records:
        by_batch[str(record.metadata["source_batch"])].append(record)
    choices = []
    for batch, batch_records in sorted(by_batch.items()):
        ordered = sorted(batch_records, key=lambda record: record.record_id)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                candidate_a, candidate_b = _outcome_blind_orientation(left, right)
                candidate_a_features = _expanded_visible_features(candidate_a)
                candidate_b_features = _expanded_visible_features(candidate_b)
                visible_names = sorted(
                    set(candidate_a_features) | set(candidate_b_features)
                )
                distance = math.sqrt(
                    sum(
                        (
                            float(candidate_a_features.get(name, 0.0))
                            - float(candidate_b_features.get(name, 0.0))
                        )
                        ** 2
                        for name in visible_names
                    )
                )
                choices.append(
                    ActionRecord(
                        record_id=(
                            f"matbot-lfp-choice::{batch}::"
                            f"{_slug(candidate_a.metadata['pair_id'])}::"
                            f"{_slug(candidate_b.metadata['pair_id'])}"
                        ),
                        benchmark=CHOICE_TASK,
                        split="development_replay",
                        visible_context=(
                            "Retrospective MatBot action choice. Outcomes are withheld.\n\n"
                            f"Candidate A:\n{candidate_a.visible_context}\n\n"
                            f"Candidate B:\n{candidate_b.visible_context}"
                        ),
                        candidate_action="Execute candidate A rather than candidate B.",
                        action_type="choose_candidate",
                        evidence=[
                            f"source_batch={batch}",
                            "candidate_outcomes=withheld_until_evaluation",
                            "candidate_orientation=stable_hash_without_outcomes",
                        ],
                        features={
                            "candidate_visible_feature_distance": distance,
                            "candidate_cost_difference": (
                                candidate_a.features["cost"] - candidate_b.features["cost"]
                            ),
                            **{
                                f"candidate_delta::{name}": (
                                    float(candidate_a_features.get(name, 0.0))
                                    - float(candidate_b_features.get(name, 0.0))
                                )
                                for name in visible_names
                                if name.startswith("component_fraction::")
                                or name
                                in {
                                    "composition_entropy",
                                    "salt_fraction",
                                    "additive_fraction",
                                }
                            },
                            "source_reliability": min(
                                candidate_a.features["source_reliability"],
                                candidate_b.features["source_reliability"],
                            ),
                            "evidence_support": min(
                                candidate_a.features["evidence_support"],
                                candidate_b.features["evidence_support"],
                            ),
                            "protocol_comparability": 1.0,
                            "cost": candidate_a.features["cost"],
                            "reversibility": candidate_a.features["reversibility"],
                        },
                        label=int(candidate_a.utility > candidate_b.utility),
                        utility=candidate_a.utility,
                        metadata={
                            "source": "real_matbot_historical_pairwise_replay",
                            "source_batch": batch,
                            "candidate_a_record_id": candidate_a.record_id,
                            "candidate_b_record_id": candidate_b.record_id,
                            "hidden_outcome_for_evaluation_only": {
                                "candidate_a_utility": candidate_a.utility,
                                "candidate_b_utility": candidate_b.utility,
                                "utility_difference": (
                                    candidate_a.utility - candidate_b.utility
                                ),
                                "candidate_a_measurement": candidate_a.metadata[
                                    "hidden_outcome_for_evaluation_only"
                                ],
                                "candidate_b_measurement": candidate_b.metadata[
                                    "hidden_outcome_for_evaluation_only"
                                ],
                            },
                            "orientation_uses_hidden_outcomes": False,
                            "confirmation_status": (
                                "development_only_source_inspected_before_protocol_lock"
                            ),
                        },
                    )
                )
    return choices


def _outcome(row: dict[str, str]) -> dict[str, float | str | None]:
    if row["source_batch"].startswith("batch_3_ABA"):
        primary_name = "-15C DCR at 50% SOC, 1C, 10s"
        primary_dcr = _required_float(row, "-15℃ DCR（50%SOC，1C放电10s）_num")
    else:
        primary_name = "converted -20C DCR"
        primary_dcr = _required_float(row, "-20℃ DCR（折算）_num")
    return {
        "primary_dcr_name": primary_name,
        "primary_dcr": primary_dcr,
        "cycle_600": _optional_float(row.get("45℃循环600圈_num")),
        "conductivity_ms_cm": _optional_float(row.get("全配方电导率/mS/cm_num")),
    }


def _composition(row: dict[str, str]) -> dict[str, float]:
    return {
        component: _optional_float(row.get(f"weighted_{component}")) or 0.0
        for component in COMPONENTS
    }


def _visible_context(row: dict[str, str], composition: dict[str, float]) -> str:
    nonzero = ", ".join(
        f"{component}={value:.3f}%" for component, value in composition.items() if value > 0.0
    )
    if row["source_batch"].startswith("batch_3_ABA"):
        protocol = "LFP/graphite; -15C multi-SOC DCR and 25C DCR protocol"
    else:
        protocol = "LFP/graphite; converted -20C DCR and 45C cycling protocol"
    return (
        f"Historical MatBot retrospective action. Pair ID: {row['pair_id']}. "
        f"Registered protocol: {protocol}. Weighted full-cell formulation: {nonzero}. "
        "All measured performance values are hidden until evaluation."
    )


def _visible_features(row: dict[str, str], composition: dict[str, float]) -> dict[str, float]:
    total = sum(composition.values())
    probabilities = [value / total for value in composition.values() if value > 0.0 and total > 0.0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    maximum_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    source_reliability = 0.92 if row["source_batch"].startswith("batch_3_ABA") else 0.82
    return {
        **{
            f"component_fraction::{name}": composition[name] / 100.0
            for name in COMPONENTS
        },
        "composition_entropy": entropy / maximum_entropy if maximum_entropy else 0.0,
        "salt_fraction": sum(composition[name] for name in SALT_COMPONENTS) / 100.0,
        "additive_fraction": sum(composition[name] for name in ADDITIVE_COMPONENTS) / 100.0,
        "source_reliability": source_reliability,
        "evidence_support": 1.0,
        "protocol_comparability": 1.0,
        "cost": 0.75 if row["source_batch"].startswith("batch_3_ABA") else 0.65,
        "reversibility": 0.35,
    }


def _rank_score(value: float, values: list[float], *, higher_is_better: bool) -> float:
    ordered = sorted(values, reverse=higher_is_better)
    better = sum(candidate < value for candidate in ordered)
    if higher_is_better:
        better = sum(candidate > value for candidate in ordered)
    equal = sum(candidate == value for candidate in ordered)
    rank = better + 0.5 * equal
    return 1.0 - rank / len(ordered)


def _required_float(row: dict[str, str], key: str) -> float:
    value = _optional_float(row.get(key))
    if value is None:
        raise ValueError(f"missing required outcome {key} for {row.get('pair_id')}")
    return value


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _slug(value: str) -> str:
    return "-".join(value.lower().replace("+", " ").split())


def _outcome_blind_orientation(
    left: ActionRecord,
    right: ActionRecord,
) -> tuple[ActionRecord, ActionRecord]:
    import hashlib

    digest = hashlib.sha256(
        f"{left.record_id}|{right.record_id}".encode("utf-8")
    ).digest()
    return (left, right) if digest[0] % 2 == 0 else (right, left)


def _expanded_visible_features(record: ActionRecord) -> dict[str, float]:
    features = dict(record.features)
    for component, value in re.findall(r"([A-Za-z0-9\u4e00-\u9fff]+)=([0-9.]+)%", record.visible_context):
        if component in COMPONENTS:
            features.setdefault(f"component_fraction::{component}", float(value) / 100.0)
    return features
