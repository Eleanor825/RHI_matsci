import json
from pathlib import Path

import pytest

from harness_matsci.lfp_crossed_gain import (
    CrossedPairwiseGainModel,
    apply_source_fuser,
    fit_candidate_excluded_source_fuser,
)
from harness_matsci.schema import ActionRecord


def test_crossed_model_outputs_exact_source_decomposition():
    fold = Path("research/evl_iclr/experiments/lfp_real_multitask/folds/fold_0")
    train = [
        ActionRecord.from_json(json.loads(line))
        for line in (fold / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    test = [
        ActionRecord.from_json(json.loads(line))
        for line in (fold / "test.jsonl").read_text().splitlines()
        if line.strip()
    ]

    model = CrossedPairwiseGainModel(replicas=3, neighbors=3).fit(train)
    state = model.predict_one(test[0])

    assert len(state.source_matrix) == 3
    assert all(len(row) == 4 for row in state.source_matrix)
    total = sum(
        (
            value
            - sum(sum(row) for row in state.source_matrix)
            / (len(state.source_matrix) * len(state.source_matrix[0]))
        )
        ** 2
        for row in state.source_matrix
        for value in row
    ) / (len(state.source_matrix) * len(state.source_matrix[0]))
    assert abs(
        total
        - state.crossed_internal_variance
        - state.crossed_external_variance
        - state.crossed_interaction_variance
    ) < 1e-12
    assert -1.0 <= state.actual_comparative_gain <= 1.0
    assert all(-1.0 <= value <= 1.0 for row in state.source_matrix for value in row)


def test_candidate_excluded_fuser_learns_convex_source_weights():
    fold = Path("research/evl_iclr/experiments/lfp_real_multitask/folds/fold_0")
    train = [
        ActionRecord.from_json(json.loads(line))
        for line in (fold / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    test = [
        ActionRecord.from_json(json.loads(line))
        for line in (fold / "test.jsonl").read_text().splitlines()
        if line.strip()
    ]

    fuser = fit_candidate_excluded_source_fuser(train, replicas=2)
    weights = next(iter(fuser.parameters_by_task.values())).source_weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in weights.values())
    model = CrossedPairwiseGainModel(replicas=2, neighbors=2).fit(train)
    fused = apply_source_fuser(model.predict(test[:2]), fuser)
    assert len(fused) == 2
