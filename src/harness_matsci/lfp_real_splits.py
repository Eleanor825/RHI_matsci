from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .schema import ActionRecord


PRIMARY_FAMILY = "AEA"
PRIMARY_OBJECTIVES = (
    "conductivity",
    "formation_gas",
    "first_efficiency",
    "specific_capacity",
    "internal_resistance",
)


@dataclass(frozen=True)
class LfpRealFold:
    fold: int
    held_candidate_ids: tuple[str, ...]
    train_candidate_ids: tuple[str, ...]
    feedback_candidate_ids: tuple[str, ...]
    acceptance_candidate_ids: tuple[str, ...]
    train: tuple[ActionRecord, ...]
    feedback: tuple[ActionRecord, ...]
    acceptance: tuple[ActionRecord, ...]
    test: tuple[ActionRecord, ...]
    excluded_cross_edges: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "held_candidate_ids": list(self.held_candidate_ids),
            "train_candidate_ids": list(self.train_candidate_ids),
            "feedback_candidate_ids": list(self.feedback_candidate_ids),
            "acceptance_candidate_ids": list(self.acceptance_candidate_ids),
            "train_count": len(self.train),
            "feedback_count": len(self.feedback),
            "acceptance_count": len(self.acceptance),
            "test_count": len(self.test),
            "excluded_cross_edges": self.excluded_cross_edges,
        }


def build_candidate_disjoint_lfp_folds(
    records: list[ActionRecord],
    *,
    folds: int = 4,
) -> list[LfpRealFold]:
    selected = [
        record
        for record in records
        if record.metadata.get("family") == PRIMARY_FAMILY
        and record.metadata.get("objective") in PRIMARY_OBJECTIVES
    ]
    candidate_ids = sorted(
        {
            str(candidate_id)
            for record in selected
            for candidate_id in record.metadata["candidate_group_ids"]
        },
        key=lambda value: (_digest(value), value),
    )
    if len(candidate_ids) < folds * 2 or len(candidate_ids) % folds:
        raise ValueError("candidate count must be divisible into nontrivial folds")
    held_groups = tuple(
        tuple(candidate_ids[index::folds]) for index in range(folds)
    )
    output = []
    for fold, held_candidate_ids in enumerate(held_groups):
        held = set(held_candidate_ids)
        remaining = sorted(
            set(candidate_ids) - held,
            key=lambda value: (_digest(f"{fold}:{value}"), value),
        )
        train_candidates = set(remaining[:6])
        feedback_candidates = set(remaining[6:9])
        acceptance_candidates = set(remaining[9:])
        split_candidates = {
            "train": train_candidates,
            "feedback": feedback_candidates,
            "acceptance": acceptance_candidates,
            "test": held,
        }
        partitions = {split: [] for split in split_candidates}
        excluded = 0
        for record in selected:
            candidate_pair = set(record.metadata["candidate_group_ids"])
            matched = [
                split
                for split, candidates in split_candidates.items()
                if candidate_pair <= candidates
            ]
            if len(matched) == 1:
                partitions[matched[0]].append(record)
            else:
                excluded += 1
        output.append(
            LfpRealFold(
                fold=fold,
                held_candidate_ids=tuple(sorted(held_candidate_ids)),
                train_candidate_ids=tuple(sorted(train_candidates)),
                feedback_candidate_ids=tuple(sorted(feedback_candidates)),
                acceptance_candidate_ids=tuple(sorted(acceptance_candidates)),
                train=tuple(sorted(partitions["train"], key=lambda row: row.record_id)),
                feedback=tuple(sorted(partitions["feedback"], key=lambda row: row.record_id)),
                acceptance=tuple(sorted(partitions["acceptance"], key=lambda row: row.record_id)),
                test=tuple(sorted(partitions["test"], key=lambda row: row.record_id)),
                excluded_cross_edges=excluded,
            )
        )
    return output


def audit_candidate_disjoint_fold(fold: LfpRealFold) -> dict[str, object]:
    split_records = {
        "train": fold.train,
        "feedback": fold.feedback,
        "acceptance": fold.acceptance,
        "test": fold.test,
    }
    record_ids = {
        split: {record.record_id for record in records}
        for split, records in split_records.items()
    }
    candidates_by_split = {
        split: {
            candidate
            for record in records
            for candidate in record.metadata["candidate_group_ids"]
        }
        for split, records in split_records.items()
    }
    record_overlap = {
        f"{left}::{right}": len(record_ids[left] & record_ids[right])
        for left_index, left in enumerate(record_ids)
        for right in tuple(record_ids)[left_index + 1 :]
    }
    tasks = {
        split: sorted({record.metadata["objective"] for record in records})
        for split, records in split_records.items()
    }
    candidate_overlap = {
        f"{left}::{right}": sorted(candidates_by_split[left] & candidates_by_split[right])
        for left_index, left in enumerate(candidates_by_split)
        for right in tuple(candidates_by_split)[left_index + 1 :]
    }
    expected_candidates = {
        "train": set(fold.train_candidate_ids),
        "feedback": set(fold.feedback_candidate_ids),
        "acceptance": set(fold.acceptance_candidate_ids),
        "test": set(fold.held_candidate_ids),
    }
    passed = (
        not any(record_overlap.values())
        and not any(candidate_overlap.values())
        and all(candidates_by_split[split] == expected for split, expected in expected_candidates.items())
        and all(set(values) == set(PRIMARY_OBJECTIVES) for values in tasks.values())
    )
    return {
        "passed": passed,
        "record_overlap": record_overlap,
        "candidate_counts": {
            split: len(candidates) for split, candidates in candidates_by_split.items()
        },
        "candidate_overlap": candidate_overlap,
        "tasks": tasks,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
