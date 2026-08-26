import json
from pathlib import Path

from harness_matsci.lfp_choice_views import LfpChoiceEvidenceViewBuilder
from harness_matsci.schema import ActionRecord


def _records():
    path = Path(
        "research/evl_iclr/experiments/matbot_lfp_historical_replay/"
        "pairwise_choices.jsonl"
    )
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_builds_balanced_outcome_blind_external_views():
    records = _records()
    held_batch = records[0].metadata["source_batch"]
    held = next(
        record for record in records if record.metadata["source_batch"] == held_batch
    )
    train = [
        record for record in records if record.metadata["source_batch"] != held_batch
    ]

    views = LfpChoiceEvidenceViewBuilder(train, neighbors=3).views(held)

    assert [view.view_id for view in views] == [
        "evidence::base",
        "tool::formulation_descriptors",
        "tool::held_batch_analogs",
    ]
    assert len({view.record.record_id for view in views}) == 3
    descriptor_text = "\n".join(views[1].record.evidence)
    assert "component_fraction" in descriptor_text
    analog_results = views[2].provenance["results"]
    assert analog_results
    assert all(held_batch not in result["record_id"] for result in analog_results)
    assert all(view.provenance["target_outcome_hidden"] for view in views)
