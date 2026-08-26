import json
from pathlib import Path

from experiments.sd_worth.paired_numeric_ablation_significance import (
    compare_numeric_methods,
)


INPUT = Path(
    "research/evl_iclr/data/llm_same_evidence_round2/subset/"
    "adaptive_fit_acceptance_485.jsonl"
)


def test_identical_numeric_methods_have_zero_difference():
    rows = [json.loads(line) for line in INPUT.read_text().splitlines()]
    acceptance = [row for row in rows if row["split"] == "acceptance"][:10]
    record_ids = [row["record_id"] for row in acceptance]
    method = {
        "selected_record_ids": record_ids[:2],
        "stage_by_record_id": {record_id: "base" for record_id in record_ids},
    }
    result = {"summary": {"acceptance": {"a": method, "b": method}}}
    audit = compare_numeric_methods(
        result,
        acceptance,
        primary_method="a",
        baseline_method="b",
        draws=100,
        seed=7,
    )
    assert audit["paired_gain_difference"] == 0.0
    assert not audit["superiority_passed"]
