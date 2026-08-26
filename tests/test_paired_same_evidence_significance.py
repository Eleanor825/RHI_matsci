import json
from pathlib import Path

from experiments.sd_worth.paired_same_evidence_significance import compare
from harness_matsci.adaptive_evidence_dataset import AdaptiveEvidenceExample


INPUT = Path(
    "research/evl_iclr/data/llm_same_evidence_round2/subset/"
    "adaptive_fit_acceptance_485.jsonl"
)


def test_paired_audit_reconstructs_same_record_values():
    rows = [json.loads(line) for line in INPUT.read_text().splitlines()]
    acceptance = [row for row in rows if row["split"] == "acceptance"][:12]
    examples = [AdaptiveEvidenceExample(row) for row in acceptance]
    selected = [example.record_id for example in examples[:3]]
    numeric = {
        "summary": {
            "acceptance": {
                "numeric": {
                    "selected_record_ids": selected,
                    "stage_by_record_id": {
                        example.record_id: "base" for example in examples
                    },
                    "selective_risk": 0.0,
                    "risk_upper_bound": 0.1,
                }
            }
        }
    }
    llm = {
        "results": {
            "llm": {
                "selected_record_ids": [],
                "selective_risk": 0.0,
                "risk_upper_bound": 0.2,
            }
        }
    }
    llm_rows = [
        {
            "record_id": example.record_id,
            "target_gain": example.realized_gain("base"),
            "acquired_evidence_cost": 0.0,
        }
        for example in examples
    ]
    result = compare(
        numeric,
        llm,
        acceptance,
        llm_rows,
        numeric_method="numeric",
        llm_method="llm",
        draws=100,
        seed=7,
    )
    assert result["record_count"] == 12
    assert result["numeric_total_strict_gain"] == sum(
        example.realized_gain("base") for example in examples[:3]
    )
