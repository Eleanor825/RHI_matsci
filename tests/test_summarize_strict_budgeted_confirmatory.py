import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "sd_worth"
    / "summarize_strict_budgeted_confirmatory.py"
)
SPEC = importlib.util.spec_from_file_location("strict_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stratified_summary_compares_frozen_global_policies():
    rows = []
    for index in range(20):
        gain = 0.8 if index < 4 else -0.2
        rows.append(
            {
                "benchmark": "a" if index % 2 == 0 else "b",
                "actual_strict_net_gain": gain,
                "scores": {
                    "factorized_conformal_lower_gain": 0.9 - index / 100,
                    "factorized_expected_gain": 0.8 - index / 100,
                    "direct_llm_judge": index / 20,
                    "llm_component_gain": index / 20,
                    "static_reliability": index / 20,
                },
            }
        )
    policies = {
        "factorized_conformal_lower_gain": {
            "budget_fraction": 0.1,
            "minimum_score": 0.0,
            "scope": "global",
        },
        "factorized_expected_gain": {
            "budget_fraction": 0.1,
            "minimum_score": 0.0,
            "scope": "global",
        },
        "direct_llm_judge": {
            "budget_fraction": 0.1,
            "minimum_score": 0.5,
            "scope": "global",
        },
        "llm_component_gain": {
            "budget_fraction": 0.1,
            "minimum_score": 0.0,
            "scope": "global",
        },
        "static_reliability": {
            "budget_fraction": 0.1,
            "minimum_score": 0.5,
            "scope": "global",
        },
    }

    report = MODULE.summarize(
        {"test_predictions": rows, "policies": policies},
        samples=100,
        seed=7,
    )

    assert report["observed"]["factorized_conformal_lower_gain"]["selective_risk"] == 0.0
    assert report["observed"]["direct_llm_judge"]["selective_risk"] == 1.0
    assert "total_gain_advantage" in report["comparisons"]["direct_llm_judge"]
    ours_by_task = report["task_observed"]["factorized_conformal_lower_gain"]
    assert sum(row["selected_count"] for row in ours_by_task.values()) == 2
    assert sum(row["selection_share"] for row in ours_by_task.values()) == 1.0
    assert all("coverage" in row for row in ours_by_task.values())


def test_task_breakdown_marks_unselected_task_risk_as_missing():
    rows = [
        {
            "benchmark": "a",
            "actual_strict_net_gain": 0.8,
            "scores": {"ours": 0.9},
        },
        {
            "benchmark": "b",
            "actual_strict_net_gain": -0.2,
            "scores": {"ours": -0.1},
        },
    ]
    policy = MODULE.BudgetedGainPolicy(
        budget_fraction=0.5,
        minimum_score=0.0,
    )

    report = MODULE._task_breakdown(rows, "ours", policy)

    assert report["a"]["selected_count"] == 1
    assert report["a"]["selective_risk"] == 0.0
    assert report["b"]["selected_count"] == 0
    assert report["b"]["selective_risk"] is None
    assert report["b"]["selected_mean_net_gain"] is None
