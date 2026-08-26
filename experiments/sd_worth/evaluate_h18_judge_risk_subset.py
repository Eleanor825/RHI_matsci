from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.hurdle_gain_calibration import HurdleGainPrediction
from harness_matsci.two_stage_policy import (
    evaluate_two_stage_policy,
    select_two_stage_policy,
)


COVERAGES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
TOOL_COST = 0.020


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--judge-cache", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(
        args.predictions,
        args.sample_manifest,
        judge_caches=args.judge_cache,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")


def evaluate(predictions_path, sample_manifest_path, *, judge_caches):
    prediction_result = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(sample_manifest_path).read_text(encoding="utf-8"))
    selected_ids = {
        split: set(values) for split, values in manifest["record_ids_by_split"].items()
    }
    rows = {
        split: [
            row
            for row in prediction_result["record_predictions"][split]
            if row["record_id"] in selected_ids[split]
        ]
        for split in ("feedback", "acceptance", "test")
    }
    for split in rows:
        if len(rows[split]) != len(selected_ids[split]):
            raise ValueError(f"missing predictions for {split}")
    score_maps = {
        "ours_source_voi_probability": {
            split: {
                row["record_id"]: row["acquisition_scores"]["source_voi_probability"]
                for row in rows[split]
            }
            for split in rows
        },
        "matched_no_source_voi_probability": {
            split: {
                row["record_id"]: row["acquisition_scores"]["no_source_voi_probability"]
                for row in rows[split]
            }
            for split in rows
        },
        "static_external_variance": {
            split: {
                row["record_id"]: row["base_external_variance"]
                for row in rows[split]
            }
            for split in rows
        },
    }
    for cache_path in judge_caches:
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        model = cache["model"]
        scores = cache["scores"]
        score_maps[f"llm_direct_{model}"] = {
            split: {row["record_id"]: scores[row["record_id"]] for row in rows[split]}
            for split in rows
        }
    methods = {}
    for method, by_split in score_maps.items():
        policy = select_two_stage_policy(
            _aligned_scores(rows["feedback"], by_split["feedback"]),
            [_prediction(row, "base_tail_risk_score") for row in rows["feedback"]],
            [_prediction(row, "tool_tail_risk_score") for row in rows["feedback"]],
            [row["label"] for row in rows["feedback"]],
            [row["gain"] for row in rows["feedback"]],
            tool_cost=TOOL_COST,
            coverages=COVERAGES,
            base_execution_coverages=(0.0,),
            tool_execution_coverages=COVERAGES,
            max_empirical_risk=0.20,
        )
        acceptance = _evaluate(policy, rows["acceptance"], by_split["acceptance"])
        test = _evaluate(policy, rows["test"], by_split["test"])
        methods[method] = {
            "feedback_policy": policy.to_json(),
            "acceptance_certificate": acceptance.to_json(),
            "test_evaluation": test.to_json(),
            "deployed": acceptance.certified,
            "deployed_test_population_gain": (
                test.population_gain_after_cost if acceptance.certified else 0.0
            ),
        }
    return {
        "schema": "h18-risk-certified-judge-subset-v1",
        "sample_manifest": manifest,
        "tool_cost": TOOL_COST,
        "alpha": 0.20,
        "delta": 0.05,
        "methods": methods,
    }


def _evaluate(policy, rows, score_map):
    return evaluate_two_stage_policy(
        policy,
        _aligned_scores(rows, score_map),
        [_prediction(row, "base_tail_risk_score") for row in rows],
        [_prediction(row, "tool_tail_risk_score") for row in rows],
        [row["label"] for row in rows],
        [row["gain"] for row in rows],
        tool_cost=TOOL_COST,
        alpha=0.20,
        delta=0.05,
    )


def _aligned_scores(rows, score_map):
    return [float(score_map[row["record_id"]]) for row in rows]


def _prediction(row, field):
    score = float(row[field])
    return HurdleGainPrediction(
        record_id=row["record_id"],
        benchmark="external_pairwise_mixed_dielectric",
        mean_gain=score,
        p_positive_gain=float(row["base_p_positive_gain"]),
        expected_shortfall=0.0,
        tail_risk_score=score,
        irreducible_variance=0.0,
        calibrated_internal_variance=float(row["base_internal_variance"]),
        calibrated_external_variance=float(row["base_external_variance"]),
        calibrated_variance=(
            float(row["base_internal_variance"])
            + float(row["base_external_variance"])
        ),
        positive_component_mean=max(0.0, score),
        negative_component_mean=min(0.0, score),
    )


def markdown(result):
    lines = [
        "# H18 Risk-Certified LLM Judge Subset",
        "",
        "Each split contains 500 actions sampled without hidden labels. All methods use the same feedback policy selection, acceptance certificate, tool cost, and verify-before-execute contract.",
        "",
        "| method | acceptance acquired | acceptance executed | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for method, row in result["methods"].items():
        acceptance = row["acceptance_certificate"]
        test = row["test_evaluation"]
        lines.append(
            f"| `{method}` | {acceptance['acquired_count']} | "
            f"{acceptance['executed_count']} | {acceptance['risk_upper_bound']:.4f} | "
            f"`{row['deployed']}` | {test['acquired_count']} | "
            f"{test['executed_count']} | {test['selective_risk']:.4f} | "
            f"{row['deployed_test_population_gain']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
