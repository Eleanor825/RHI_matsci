import csv
from pathlib import Path

import pytest

from harness_matsci.lfp_formulation_database import (
    build_lfp_candidate_actions,
    build_lfp_pairwise_actions,
    load_lfp_formulation_experiments,
)


def test_builds_weighted_real_formulations_and_outcome_blind_pairs(tmp_path: Path):
    path = tmp_path / "lfp.csv"
    fields = [
        "牌号",
        "注液比例",
        "EC",
        "DMC",
        "LiPF6",
        "电导率",
        "电芯下线首效",
        "电芯下线放电材料比容量",
        "电芯下线内阻",
    ]
    rows = []
    for index, efficiency in enumerate((90.0, 91.0, 92.0), 1):
        rows.extend(
            [
                [f"AEA{index:02d}A", "80", "30", "60", "10", str(10 + index), str(efficiency), str(130 + index), str(10 - index)],
                [f"AEA{index:02d}B", "20", "20", "70", "10", str(12 + index), "", "", ""],
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    experiments = load_lfp_formulation_experiments(path)
    candidates = build_lfp_candidate_actions(experiments)
    pairs = build_lfp_pairwise_actions(candidates)

    assert len(experiments) == 3
    assert experiments[0].composition["EC"] == 28.0
    efficiency_candidates = [
        row for row in candidates if row.benchmark == "lfp_real_first_efficiency"
    ]
    assert len(efficiency_candidates) == 3
    assert sorted(row.utility for row in efficiency_candidates) == pytest.approx(
        [1 / 6, 0.5, 5 / 6]
    )
    efficiency_pairs = [
        row for row in pairs if row.benchmark == "lfp_real_pair_first_efficiency"
    ]
    assert len(efficiency_pairs) == 3
    assert all(not row.metadata["orientation_uses_hidden_outcomes"] for row in efficiency_pairs)
    assert all("raw_measurement" not in row.visible_context for row in candidates)
