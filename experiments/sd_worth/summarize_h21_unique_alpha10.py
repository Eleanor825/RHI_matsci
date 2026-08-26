from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import beta


PRIMARY = "source_voi_expected"
COMPARATORS = ("no_source_voi_expected", "external_variance")
EVOLUTION_SEQUENCE = (
    "external_variance",
    "no_source_voi_expected",
    "source_voi_expected",
    "source_voi_probability",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", default="1,2,3,4")
    parser.add_argument("--benchmark-root")
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",") if value]
    reports = [
        json.loads(
            (Path(args.results_root) / f"fold{fold}" / "result.json").read_text()
        )
        for fold in folds
    ]
    test_counts = None
    if args.benchmark_root:
        benchmark_root = Path(args.benchmark_root)
        test_counts = {
            fold: int(
                json.loads(
                    (benchmark_root / f"fold_{fold}" / "manifest.json").read_text()
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


def summarize(
    reports: list[dict],
    *,
    folds: list[int],
    test_counts: dict[int, int] | None = None,
) -> dict:
    if len(reports) != len(folds):
        raise ValueError("fold labels and reports must have equal length")
    methods = (PRIMARY, *COMPARATORS)
    aggregates = {
        method: _aggregate_method(
            reports,
            folds=folds,
            method=method,
            test_counts=test_counts,
        )
        for method in methods
    }
    primary = aggregates[PRIMARY]
    evolved = _aggregate_evolution(reports, folds=folds, test_counts=test_counts)
    gate = {
        "certified_in_at_least_three_folds": primary["certified_fold_count"] >= 3,
        "positive_aggregate_deployed_test_gain": primary[
            "pooled_population_gain_after_cost"
        ]
        > 0.0,
        "pooled_empirical_test_risk_at_most_alpha": primary[
            "pooled_selective_risk"
        ]
        <= 0.10,
        "beats_external_variance_gain": primary[
            "pooled_population_gain_after_cost"
        ]
        > aggregates["external_variance"]["pooled_population_gain_after_cost"],
    }
    gate["passed"] = all(gate.values())
    evolution_gate = {
        "deploys_in_at_least_three_folds": evolved["deployed_fold_count"] >= 3,
        "positive_pooled_test_gain": evolved["pooled_population_gain_after_cost"]
        > 0.0,
        "pooled_empirical_test_risk_at_most_alpha": evolved[
            "pooled_selective_risk"
        ]
        <= 0.10,
        "beats_external_variance_gain": evolved[
            "pooled_population_gain_after_cost"
        ]
        > aggregates["external_variance"]["pooled_population_gain_after_cost"],
    }
    evolution_gate["passed"] = all(evolution_gate.values())
    return {
        "schema_version": "h21-unique-alpha10-confirmatory-v1",
        "folds": folds,
        "primary_method": PRIMARY,
        "alpha": 0.10,
        "delta": 0.05,
        "methods": aggregates,
        "evolved_contract": evolved,
        "source_ablation_gain_delta": primary["pooled_population_gain_after_cost"]
        - aggregates["no_source_voi_expected"]["pooled_population_gain_after_cost"],
        "confirmatory_gate": gate,
        "evolution_gate": evolution_gate,
    }


def _aggregate_evolution(
    reports: list[dict],
    *,
    folds: list[int],
    test_counts: dict[int, int] | None,
) -> dict:
    rows = [
        _evolve_fold(
            report,
            fold=fold,
            test_count=(test_counts or {}).get(fold),
        )
        for fold, report in zip(folds, reports)
    ]
    population = sum(row["test_count"] for row in rows)
    executed = sum(row["test_executed"] for row in rows)
    errors = sum(row["test_errors"] for row in rows)
    total_gain = sum(row["test_population_gain"] * row["test_count"] for row in rows)
    output = {
        "candidate_sequence": list(EVOLUTION_SEQUENCE),
        "familywise_delta": 0.05,
        "candidate_delta": 0.05 / len(EVOLUTION_SEQUENCE),
        "deployed_fold_count": sum(row["selected_contract"] != "no_op" for row in rows),
        "source_feature_selected_fold_count": sum(
            row["selected_contract"].startswith("source_") for row in rows
        ),
        "pooled_test_count": population,
        "pooled_executed_count": executed,
        "pooled_error_count": errors,
        "pooled_selective_risk": errors / executed if executed else 0.0,
        "pooled_risk_upper_bound_95": _cp_upper(errors, executed),
        "pooled_population_gain_after_cost": total_gain / population if population else 0.0,
        "folds": rows,
    }
    return output


def _evolve_fold(report: dict, *, fold: int, test_count: int | None) -> dict:
    incumbent = "no_op"
    incumbent_gain = 0.0
    trace = []
    candidate_delta = 0.05 / len(EVOLUTION_SEQUENCE)
    for cycle, method in enumerate(EVOLUTION_SEQUENCE):
        values = report["policies"][method]
        acceptance = values["acceptance_certificate"]
        executed = int(acceptance["executed_count"])
        errors = int(acceptance["errors"])
        risk_upper_bound = _cp_upper(errors, executed, delta=candidate_delta)
        gain = float(acceptance["population_gain_after_cost"])
        feasible = executed > 0 and risk_upper_bound <= 0.10
        accepted = feasible and gain > incumbent_gain + 1e-6
        if accepted:
            incumbent = method
            incumbent_gain = gain
        trace.append(
            {
                "cycle": cycle,
                "candidate": method,
                "acceptance_executed": executed,
                "acceptance_errors": errors,
                "familywise_risk_upper_bound": risk_upper_bound,
                "acceptance_population_gain": gain,
                "feasible": feasible,
                "accepted": accepted,
                "incumbent_after_cycle": incumbent,
            }
        )
    if incumbent == "no_op":
        population = test_count or _report_test_population(report)
        return {
            "fold": fold,
            "selected_contract": incumbent,
            "test_count": population,
            "test_executed": 0,
            "test_errors": 0,
            "test_selective_risk": 0.0,
            "test_population_gain": 0.0,
            "trace": trace,
        }
    test = report["policies"][incumbent]["test_evaluation"]
    return {
        "fold": fold,
        "selected_contract": incumbent,
        "test_count": _population_count(test),
        "test_executed": int(test["executed_count"]),
        "test_errors": int(test["errors"]),
        "test_selective_risk": float(test["selective_risk"]),
        "test_population_gain": float(test["population_gain_after_cost"]),
        "trace": trace,
    }


def _report_test_population(report: dict) -> int:
    for method in EVOLUTION_SEQUENCE:
        test = report["policies"][method]["test_evaluation"]
        try:
            return _population_count(test)
        except ValueError:
            continue
    raise ValueError("cannot infer test population for no-op evolved fold")


def _aggregate_method(
    reports: list[dict],
    *,
    folds: list[int],
    method: str,
    test_counts: dict[int, int] | None,
) -> dict:
    rows = []
    for fold, report in zip(folds, reports):
        values = report["policies"][method]
        test = values["test_evaluation"]
        try:
            test_count = _population_count(test)
        except ValueError:
            test_count = (test_counts or {}).get(fold) or _report_test_population(report)
        deployed = bool(values["deployed"])
        rows.append(
            {
                "fold": fold,
                "acceptance_certified": deployed,
                "test_count": test_count,
                "test_executed": int(test["executed_count"]) if deployed else 0,
                "test_errors": int(test["errors"]) if deployed else 0,
                "test_selective_risk": float(test["selective_risk"]) if deployed else 0.0,
                "deployed_test_population_gain": float(
                    values["deployed_test_population_gain"]
                ),
            }
        )
    population = sum(row["test_count"] for row in rows)
    executed = sum(row["test_executed"] for row in rows)
    errors = sum(row["test_errors"] for row in rows)
    total_gain = sum(
        row["deployed_test_population_gain"] * row["test_count"] for row in rows
    )
    return {
        "certified_fold_count": sum(row["acceptance_certified"] for row in rows),
        "pooled_test_count": population,
        "pooled_executed_count": executed,
        "pooled_error_count": errors,
        "pooled_selective_risk": errors / executed if executed else 0.0,
        "pooled_risk_upper_bound_95": _cp_upper(errors, executed),
        "pooled_population_gain_after_cost": total_gain / population if population else 0.0,
        "folds": rows,
    }


def _population_count(test: dict) -> int:
    coverage = float(test["execution_coverage"])
    executed = int(test["executed_count"])
    if coverage > 0.0:
        return int(round(executed / coverage))
    acquisition_coverage = float(test["acquisition_coverage"])
    acquired = int(test["acquired_count"])
    if acquisition_coverage > 0.0:
        return int(round(acquired / acquisition_coverage))
    raise ValueError("cannot infer test population from zero-coverage evaluation")


def _cp_upper(errors: int, total: int, delta: float = 0.05) -> float:
    if total == 0:
        return 0.0
    if errors == total:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


def markdown(summary: dict) -> str:
    lines = [
        "# H21 Unique-Material Alpha-0.10 Confirmatory Results",
        "",
        "| method | certified folds | executed | errors | pooled risk | risk UCB | population gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        lines.append(
            f"| `{method}` | {values['certified_fold_count']}/4 | "
            f"{values['pooled_executed_count']} | {values['pooled_error_count']} | "
            f"{values['pooled_selective_risk']:.4f} | "
            f"{values['pooled_risk_upper_bound_95']:.4f} | "
            f"{values['pooled_population_gain_after_cost']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Source-ablation gain delta: `{summary['source_ablation_gain_delta']:+.6f}`.",
            f"Confirmatory gate passed: `{summary['confirmatory_gate']['passed']}`.",
            "",
            "## Recursive Signal Evolution",
            "",
            f"Selected contracts by fold: `{[row['selected_contract'] for row in summary['evolved_contract']['folds']]}`.",
            f"Pooled evolved test gain: `{summary['evolved_contract']['pooled_population_gain_after_cost']:+.6f}`.",
            f"Pooled evolved risk: `{summary['evolved_contract']['pooled_selective_risk']:.4f}`.",
            f"Evolution gate passed: `{summary['evolution_gate']['passed']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
