from harness_matsci.external_h23_superconductivity import (
    TASK,
    build_external_h23_fold,
    load_external_h23_materials,
)
from harness_matsci.external_h23_superconductivity_tools import (
    ExternalH23SuperconductivityEnvironment,
)


SOURCE = "benchmarks/external_h23_superconductivity/raw/superconductivity2018.json.gz"


def test_loads_superconductivity_records():
    rows = load_external_h23_materials(SOURCE)
    assert len(rows) == 16406
    assert {row.row_id for row in rows}.isdisjoint(
        {2095, 4554, 9806, 9848, 13547, 14018, 14504, 16346}
    )
    assert len(rows[0].vector) == 130


def test_group_disjoint_superconductivity_fold():
    fold = build_external_h23_fold(SOURCE, fold=0)
    groups = {split: set(values) for split, values in fold.groups_by_split.items()}
    assert not groups["train"] & groups["test"]
    assert all(row.benchmark == TASK for row in fold.records_by_split["test"])
    assert all("Tc=" not in row.visible_context for row in fold.records_by_split["test"])


def test_high_fidelity_tc_view_preserves_hidden_outcome_boundary():
    fold = build_external_h23_fold(SOURCE, fold=0)
    environment = ExternalH23SuperconductivityEnvironment.fit(
        SOURCE,
        fold.records_by_split["train"],
        seed=2301,
    )
    view = environment.observe(fold.records_by_split["test"][0], source="high_fidelity")
    assert view.metadata["hidden_outcome_exposed"] is False
    assert view.metadata["outcome_defining_measurement_replayed"] is True
    assert "tool_estimated_utility" in view.features
