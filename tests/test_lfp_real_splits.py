import json
from pathlib import Path

from harness_matsci.lfp_real_splits import (
    PRIMARY_OBJECTIVES,
    audit_candidate_disjoint_fold,
    build_candidate_disjoint_lfp_folds,
)
from harness_matsci.schema import ActionRecord


def test_builds_candidate_disjoint_multitask_folds():
    path = Path("research/evl_iclr/data/lfp_real_database_v1/pairwise_actions.jsonl")
    records = [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    folds = build_candidate_disjoint_lfp_folds(records, folds=4)

    assert len(folds) == 4
    assert all(audit_candidate_disjoint_fold(fold)["passed"] for fold in folds)
    assert all(len(fold.held_candidate_ids) == 5 for fold in folds)
    assert all(len(fold.train_candidate_ids) == 6 for fold in folds)
    assert all(len(fold.feedback_candidate_ids) == 3 for fold in folds)
    assert all(len(fold.acceptance_candidate_ids) == 6 for fold in folds)
    assert all(len(fold.test) == 10 * len(PRIMARY_OBJECTIVES) for fold in folds)
    assert all(len(fold.train) == 15 * len(PRIMARY_OBJECTIVES) for fold in folds)
    assert all(len(fold.feedback) == 3 * len(PRIMARY_OBJECTIVES) for fold in folds)
    assert all(len(fold.acceptance) == 15 * len(PRIMARY_OBJECTIVES) for fold in folds)
