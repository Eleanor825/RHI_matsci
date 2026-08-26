from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from harness_matsci.adaptive_evidence_dataset import AdaptiveEvidenceExample


COST_WEIGHT = 0.15


def compare(
    numeric_result: dict[str, object],
    llm_result: dict[str, object],
    adaptive_rows: list[dict[str, object]],
    llm_rows: list[dict[str, object]],
    *,
    numeric_method: str,
    llm_method: str,
    draws: int,
    seed: int,
) -> dict[str, object]:
    numeric = numeric_result["summary"]["acceptance"][numeric_method]
    llm = llm_result["results"][llm_method]
    examples = {
        example.record_id: example
        for example in (AdaptiveEvidenceExample(row) for row in adaptive_rows)
        if example.split == "acceptance"
    }
    rows_by_id = {str(row["record_id"]): row for row in llm_rows}
    if set(examples) != set(rows_by_id):
        raise ValueError("numeric and LLM acceptance records must match exactly")
    numeric_selected = set(numeric["selected_record_ids"])
    stages = numeric["stage_by_record_id"]
    llm_selected = set(llm["selected_record_ids"])
    differences = []
    by_task: dict[str, list[float]] = {}
    numeric_total = 0.0
    llm_total = 0.0
    for record_id, example in examples.items():
        stage = stages[record_id]
        numeric_value = (
            example.realized_gain(stage)
            if record_id in numeric_selected
            else -COST_WEIGHT * example.cumulative_evidence_cost(stage)
        )
        llm_row = rows_by_id[record_id]
        llm_value = (
            float(llm_row["target_gain"])
            if record_id in llm_selected
            else -COST_WEIGHT * float(llm_row["acquired_evidence_cost"])
        )
        difference = numeric_value - llm_value
        numeric_total += numeric_value
        llm_total += llm_value
        differences.append(difference)
        by_task.setdefault(example.benchmark, []).append(difference)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(sum(rng.choice(values) for _ in values) for values in by_task.values())
        for _ in range(draws)
    )
    observed = sum(differences)
    nonzero = [value for value in differences if abs(value) > 1e-12]
    randomization_exceedances = 0
    for _ in range(draws):
        randomized = sum(
            value if rng.random() < 0.5 else -value for value in nonzero
        )
        randomization_exceedances += randomized >= observed
    lower = bootstrap[int(0.025 * draws)]
    upper = bootstrap[min(draws - 1, int(0.975 * draws))]
    p_value = (randomization_exceedances + 1) / (draws + 1)
    return {
        "schema_version": "paired-same-evidence-significance-v1",
        "numeric_method": numeric_method,
        "llm_method": llm_method,
        "record_count": len(differences),
        "numeric_total_strict_gain": numeric_total,
        "llm_total_strict_gain": llm_total,
        "paired_gain_difference": observed,
        "paired_gain_difference_per_record": observed / len(differences),
        "benchmark_stratified_bootstrap_95_interval": [lower, upper],
        "one_sided_sign_flip_p_value": p_value,
        "bootstrap_draws": draws,
        "seed": seed,
        "nonzero_pair_count": len(nonzero),
        "numeric_selective_risk": numeric["selective_risk"],
        "numeric_risk_upper_bound": numeric["risk_upper_bound"],
        "llm_selective_risk": llm["selective_risk"],
        "llm_risk_upper_bound": llm["risk_upper_bound"],
        "superiority_passed": bool(lower > 0.0 and p_value < 0.05),
    }


def _jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-result", required=True)
    parser.add_argument("--llm-result", required=True)
    parser.add_argument("--adaptive-input", required=True)
    parser.add_argument("--llm-rows", required=True)
    parser.add_argument("--numeric-method", default="learned_portfolio_kg_mean")
    parser.add_argument("--llm-method", required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare(
        json.loads(Path(args.numeric_result).read_text(encoding="utf-8")),
        json.loads(Path(args.llm_result).read_text(encoding="utf-8")),
        _jsonl(args.adaptive_input),
        json.loads(Path(args.llm_rows).read_text(encoding="utf-8"))["rows"],
        numeric_method=args.numeric_method,
        llm_method=args.llm_method,
        draws=args.draws,
        seed=args.seed,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
