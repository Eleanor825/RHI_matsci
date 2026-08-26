from experiments.sd_worth.summarize_h23_superconductivity import summarize


def _method(gain, *, errors=0, deployed=True):
    return {
        "deployed": deployed,
        "deployed_test_population_gain": gain if deployed else 0.0,
        "acceptance_certificate": {
            "executed_count": 40,
            "errors": errors,
            "population_gain_after_cost": gain,
        },
        "test_evaluation": {
            "executed_count": 40,
            "errors": errors,
            "population_gain_after_cost": gain,
        },
    }


def _report():
    return {
        "policies": {
            "source_voi_expected": _method(0.004),
            "source_voi_probability": _method(0.0035),
            "no_source_voi_expected": _method(0.002),
            "external_variance": _method(0.0, deployed=False),
        }
    }


def test_primary_and_fixed_sequence_evolution_pass():
    reports = [_report() for _ in range(4)]
    summary = summarize(
        reports,
        folds=[1, 2, 3, 4],
        test_counts={1: 100, 2: 100, 3: 100, 4: 100},
    )
    assert summary["primary_gate"]["passed"]
    assert summary["evolution_gate"]["passed"]
    assert all(
        row["selected_contract"] == "source_voi_expected"
        for row in summary["evolved_contract"]["folds"]
    )
