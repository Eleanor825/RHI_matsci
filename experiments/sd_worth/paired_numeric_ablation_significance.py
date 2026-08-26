from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from harness_matsci.adaptive_evidence_dataset import AdaptiveEvidenceExample


COST_WEIGHT = 0.15


def compare_numeric_methods(
    result: dict[str, object],
    adaptive_rows: list[dict[str, object]],
    *,
    primary_method: str,
    baseline_method: str,
    draws: int,
    seed: int,
) -> dict[str, object]:
    methods = result["summary"]["acceptance"]
    primary = methods[primary_method]
    baseline = methods[baseline_method]
    examples = [
        example
        for example in (AdaptiveEvidenceExample(row) for row in adaptive_rows)
        if example.split == "acceptance"
    ]
    primary_values = _values(examples, primary)
    baseline_values = _values(examples, baseline)
    by_task: dict[str, list[float]] = {}
    differences = []
    for example in examples:
        difference = (
            primary_values[example.record_id] - baseline_values[example.record_id]
        )
        differences.append(difference)
        by_task.setdefault(example.benchmark, []).append(difference)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(sum(rng.choice(values) for _ in values) for values in by_task.values())
        for _ in range(draws)
    )
    observed = sum(differences)
    nonzero = [value for value in differences if abs(value) > 1e-12]
    exceedances = 0
    for _ in range(draws):
        randomized = sum(
            value if rng.random() < 0.5 else -value for value in nonzero
        )
        exceedances += randomized >= observed
    lower = bootstrap[int(0.025 * draws)]
    upper = bootstrap[min(draws - 1, int(0.975 * draws))]
    p_value = (exceedances + 1) / (draws + 1)
    return {
        "schema_version": "paired-numeric-ablation-significance-v1",
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "record_count": len(examples),
        "primary_total_strict_gain": sum(primary_values.values()),
        "baseline_total_strict_gain": sum(baseline_values.values()),
        "paired_gain_difference": observed,
        "benchmark_stratified_bootstrap_95_interval": [lower, upper],
        "one_sided_sign_flip_p_value": p_value,
        "nonzero_pair_count": len(nonzero),
        "bootstrap_draws": draws,
        "seed": seed,
        "superiority_passed": bool(lower > 0.0 and p_value < 0.05),
    }


def _values(examples, method):
    selected = set(method["selected_record_ids"])
    stages = method["stage_by_record_id"]
    return {
        example.record_id: (
            example.realized_gain(stages[example.record_id])
            if example.record_id in selected
            else -COST_WEIGHT
            * example.cumulative_evidence_cost(stages[example.record_id])
        )
        for example in examples
    }


def _jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--adaptive-input", required=True)
    parser.add_argument("--primary-method", default="learned_portfolio_kg_mean")
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare_numeric_methods(
        json.loads(Path(args.result).read_text(encoding="utf-8")),
        _jsonl(args.adaptive_input),
        primary_method=args.primary_method,
        baseline_method=args.baseline_method,
        draws=args.draws,
        seed=args.seed,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
