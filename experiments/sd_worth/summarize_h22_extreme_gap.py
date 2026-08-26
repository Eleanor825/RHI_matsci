from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import beta


PRIMARY = "source_voi_expected"
METHODS = (PRIMARY, "no_source_voi_expected", "external_variance")
EVOLUTION_SEQUENCE = (
    "no_source_voi_expected",
    "source_voi_expected",
    "source_voi_probability",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", default="1,2,3,4")
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",") if value]
    reports = [
        json.loads(
            (Path(args.results_root) / f"fold{fold}" / "result.json").read_text()
        )
        for fold in folds
    ]
    test_counts = {
        fold: int(
            json.loads(
                (
                    Path(args.benchmark_root)
                    / f"fold_{fold}"
                    / "manifest.json"
                ).read_text()
            )["counts"]["test"]
        )
        for fold in folds
    }
    summary = summarize(reports, folds=folds, test_counts=test_counts)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    destination.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def summarize(reports, *, folds, test_counts):
    methods = {
        method: _aggregate_method(reports, folds, test_counts, method)
        for method in METHODS
    }
    primary = methods[PRIMARY]
    primary_gate = {
        "certifies_at_least_three_folds": primary["certified_fold_count"] >= 3,
        "positive_pooled_test_gain": primary["pooled_population_gain"] > 0.0,
        "pooled_test_risk_at_most_alpha": primary["pooled_selective_risk"] <= 0.10,
        "beats_static_variance": primary["pooled_population_gain"]
        > methods["external_variance"]["pooled_population_gain"],
    }
    primary_gate["passed"] = all(primary_gate.values())
    evolved = _aggregate_evolved(reports, folds, test_counts)
    evolution_gate = {
        "deploys_at_least_three_folds": evolved["deployed_fold_count"] >= 3,
        "positive_pooled_test_gain": evolved["pooled_population_gain"] > 0.0,
        "pooled_test_risk_at_most_alpha": evolved["pooled_selective_risk"] <= 0.10,
        "beats_static_variance": evolved["pooled_population_gain"]
        > methods["external_variance"]["pooled_population_gain"],
    }
    evolution_gate["passed"] = all(evolution_gate.values())
    return {
        "schema_version": "h22-extreme-gap-confirmatory-v1",
        "folds": folds,
        "alpha": 0.10,
        "delta": 0.05,
        "primary_method": PRIMARY,
        "methods": methods,
        "primary_minus_no_source_gain": primary["pooled_population_gain"]
        - methods["no_source_voi_expected"]["pooled_population_gain"],
        "primary_gate": primary_gate,
        "evolved_contract": evolved,
        "evolution_gate": evolution_gate,
    }


def _aggregate_method(reports, folds, test_counts, method):
    rows = []
    for fold, report in zip(folds, reports):
        values = report["policies"][method]
        test = values["test_evaluation"]
        deployed = bool(values["deployed"])
        rows.append(
            {
                "fold": fold,
                "acceptance_certified": deployed,
                "test_count": test_counts[fold],
                "test_executed": int(test["executed_count"]) if deployed else 0,
                "test_errors": int(test["errors"]) if deployed else 0,
                "test_population_gain": float(
                    values["deployed_test_population_gain"]
                ),
            }
        )
    return _pool(rows, certified_key="acceptance_certified")


def _aggregate_evolved(reports, folds, test_counts):
    rows = [
        _evolve_fold(report, fold=fold, test_count=test_counts[fold])
        for fold, report in zip(folds, reports)
    ]
    output = _pool(rows, certified_key="deployed")
    output["candidate_sequence"] = list(EVOLUTION_SEQUENCE)
    output["source_feature_selected_fold_count"] = sum(
        row["selected_contract"].startswith("source_") for row in rows
    )
    return output


def _evolve_fold(report, *, fold, test_count):
    incumbent = "no_op"
    incumbent_gain = 0.0
    trace = []
    for cycle, method in enumerate(EVOLUTION_SEQUENCE):
        acceptance = report["policies"][method]["acceptance_certificate"]
        executed = int(acceptance["executed_count"])
        errors = int(acceptance["errors"])
        upper = _cp_upper(errors, executed)
        gain = float(acceptance["population_gain_after_cost"])
        certified = executed > 0 and upper <= 0.10
        accepted = certified and gain > incumbent_gain + 1e-6
        if accepted:
            incumbent = method
            incumbent_gain = gain
        trace.append(
            {
                "cycle": cycle,
                "candidate": method,
                "acceptance_executed": executed,
                "acceptance_errors": errors,
                "risk_upper_bound": upper,
                "acceptance_population_gain": gain,
                "certified": certified,
                "accepted": accepted,
                "incumbent_after_cycle": incumbent,
            }
        )
        if not certified:
            break
    if incumbent == "no_op":
        return {
            "fold": fold,
            "deployed": False,
            "selected_contract": incumbent,
            "test_count": test_count,
            "test_executed": 0,
            "test_errors": 0,
            "test_population_gain": 0.0,
            "trace": trace,
        }
    test = report["policies"][incumbent]["test_evaluation"]
    return {
        "fold": fold,
        "deployed": True,
        "selected_contract": incumbent,
        "test_count": test_count,
        "test_executed": int(test["executed_count"]),
        "test_errors": int(test["errors"]),
        "test_population_gain": float(test["population_gain_after_cost"]),
        "trace": trace,
    }


def _pool(rows, *, certified_key):
    population = sum(row["test_count"] for row in rows)
    executed = sum(row["test_executed"] for row in rows)
    errors = sum(row["test_errors"] for row in rows)
    total_gain = sum(row["test_population_gain"] * row["test_count"] for row in rows)
    return {
        "certified_fold_count": sum(bool(row[certified_key]) for row in rows),
        "deployed_fold_count": sum(bool(row[certified_key]) for row in rows),
        "pooled_test_count": population,
        "pooled_executed_count": executed,
        "pooled_error_count": errors,
        "pooled_selective_risk": errors / executed if executed else 0.0,
        "pooled_risk_upper_bound_95": _cp_upper(errors, executed),
        "pooled_population_gain": total_gain / population if population else 0.0,
        "folds": rows,
    }


def _cp_upper(errors, total, delta=0.05):
    if total == 0:
        return 0.0
    if errors == total:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


def markdown(summary):
    lines = [
        "# H22 Extreme-Gap Confirmatory Results",
        "",
        "| method | certified folds | executed | errors | risk | risk UCB | population gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        lines.append(
            f"| `{method}` | {values['certified_fold_count']}/4 | "
            f"{values['pooled_executed_count']} | {values['pooled_error_count']} | "
            f"{values['pooled_selective_risk']:.4f} | "
            f"{values['pooled_risk_upper_bound_95']:.4f} | "
            f"{values['pooled_population_gain']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Primary gate passed: `{summary['primary_gate']['passed']}`.",
            f"Primary minus no-source gain: `{summary['primary_minus_no_source_gain']:+.6f}`.",
            "",
            "## Fixed-Sequence Evolution",
            "",
            f"Selected contracts: `{[row['selected_contract'] for row in summary['evolved_contract']['folds']]}`.",
            f"Pooled evolved gain: `{summary['evolved_contract']['pooled_population_gain']:+.6f}`.",
            f"Pooled evolved risk: `{summary['evolved_contract']['pooled_selective_risk']:.4f}`.",
            f"Evolution gate passed: `{summary['evolution_gate']['passed']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
