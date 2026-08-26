import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "sd_worth"
    / "run_lfp_strict_budgeted_replay.py"
)
SPEC = importlib.util.spec_from_file_location("lfp_strict_replay", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(index):
    gain = 0.5 if index % 4 == 0 else -0.1
    return {
        "record_id": f"r{index}",
        "benchmark": "lfp",
        "fold": f"fold_{index % 5}",
        "real_scientific_outcome": True,
        "steps": [
            {
                "internal_model_replicas": [
                    {"p_proposed_action_better": 0.8 if gain > 0 else 0.2},
                    {"p_proposed_action_better": 0.7 if gain > 0 else 0.3},
                ],
                "external_tool_evidence": {
                    "fused_mean_gain": gain + 0.02,
                    "internal_variance": 0.01,
                    "external_variance": 0.01,
                    "interaction_variance": 0.005,
                },
            }
        ],
        "terminal_outcome_for_training_and_evaluation_only": {
            "realized_net_gain": gain
        },
    }


def test_runs_cross_fitted_real_outcome_replay():
    report = MODULE.run_replay([_row(index) for index in range(100)])

    assert report["trajectory_count"] == 100
    assert len(report["fold_reports"]) == 5
    assert report["summary"]["conformal_coverage"] >= 0.9
    assert report["summary"]["methods"]["factorized_expected_gain"]["selective_risk"] == 0.0
