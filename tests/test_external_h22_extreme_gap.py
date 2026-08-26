from harness_matsci.external_h22_extreme_gap import (
    TASK,
    build_external_h22_fold,
    load_external_h22_materials,
)
from harness_matsci.external_h22_extreme_gap_tools import (
    ExternalH22ExtremeGapEnvironment,
)


SOURCE = "benchmarks/external_h22_extreme_gap/raw/matbench_expt_gap.json.gz"


def test_loads_official_experimental_gap_rows():
    rows = load_external_h22_materials(SOURCE)
    assert len(rows) == 4604
    assert len(rows[0].vector) == 130


def test_group_disjoint_fold_and_hidden_outcome_contract():
    fold = build_external_h22_fold(SOURCE, fold=0)
    groups = {
        split: set(values) for split, values in fold.groups_by_split.items()
    }
    assert not groups["train"] & groups["test"]
    assert all(record.benchmark == TASK for record in fold.records_by_split["test"])
    assert all("gap expt" not in record.visible_context for record in fold.records_by_split["test"])


def test_high_fidelity_view_marks_measurement_replay_without_exposing_label():
    fold = build_external_h22_fold(SOURCE, fold=0)
    environment = ExternalH22ExtremeGapEnvironment.fit(
        SOURCE,
        fold.records_by_split["train"],
        seed=2201,
    )
    view = environment.observe(fold.records_by_split["test"][0], source="high_fidelity")
    assert view.metadata["hidden_outcome_exposed"] is False
    assert view.metadata["outcome_defining_measurement_replayed"] is True
    assert "tool_estimated_utility" in view.features
