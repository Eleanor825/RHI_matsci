from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from harness_matsci.conformal_gain import (
    ConformalGainObservation,
    fit_split_conformal_gain,
)
from harness_matsci.hybrid_product_gain import load_llm_probability_caches
from harness_matsci.lfp_crossed_gain import (
    CrossedPairwiseGainModel,
    apply_source_fuser,
    fit_candidate_excluded_source_fuser,
)
from harness_matsci.metrics import binary_metrics
from harness_matsci.risk_control import binomial_upper_confidence
from harness_matsci.safe_evidence_residual import (
    EvidenceResidualState,
    build_evidence_residual_states,
    fit_safe_aggregate_evidence_residual,
)
from harness_matsci.schema import ActionRecord


SOURCE_NAMES = ("full", "solvent_salt", "additive", "historical_analog")
THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-root", required=True)
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--calibration-cache-dir", required=True)
    parser.add_argument("--test-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        default="gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra",
    )
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1.0)
    args = parser.parse_args()
    models = tuple(value for value in args.models.split(",") if value)
    cache_paths = {
        split: {
            model: Path(directory) / f"{model}.json"
            for model in models
        }
        for split, directory in {
            "train": args.train_cache_dir,
            "feedback": args.calibration_cache_dir,
            "acceptance": args.calibration_cache_dir,
            "test": args.test_cache_dir,
        }.items()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_reports = []
    for fold_dir in sorted(Path(args.folds_root).glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        report = _run_fold(
            fold_dir,
            cache_paths=cache_paths,
            replicas=args.replicas,
            blend_weight=args.blend_weight,
            regularization=args.regularization,
        )
        fold_reports.append(report)
        (output_dir / f"{fold_dir.name}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    aggregate = _aggregate(fold_reports)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate["summary"], sort_keys=True))


def _run_fold(
    fold_dir: Path,
    *,
    cache_paths: dict[str, dict[str, Path]],
    replicas: int,
    blend_weight: float,
    regularization: float,
) -> dict[str, object]:
    splits = {
        split: _load(fold_dir / f"{split}.jsonl")
        for split in ("train", "feedback", "acceptance", "test")
    }
    fuser = fit_candidate_excluded_source_fuser(splits["train"])
    source_weights = fuser.parameters_by_task[
        "lfp_real_multitask_shared_source_fusion"
    ].source_weights
    value_model = CrossedPairwiseGainModel(replicas=replicas).fit(splits["train"])
    raw_states = {
        "train": apply_source_fuser(
            _candidate_excluded_train_states(splits["train"], replicas=replicas),
            fuser,
        ),
        **{
            split: apply_source_fuser(value_model.predict(splits[split]), fuser)
            for split in ("feedback", "acceptance", "test")
        },
    }
    states = {}
    for split, records in splits.items():
        probabilities = load_llm_probability_caches(
            records,
            model_cache_paths=cache_paths[split],
        )
        states[split] = build_evidence_residual_states(
            records,
            [state.to_json() for state in raw_states[split]],
            probabilities,
            source_names=SOURCE_NAMES,
            source_weights=source_weights,
        )
    model = fit_safe_aggregate_evidence_residual(
        states["train"],
        blend_weight=blend_weight,
        regularization=regularization,
    )
    predictions = {
        split: model.predict(states[split])
        for split in ("feedback", "acceptance", "test")
    }
    conformal = fit_split_conformal_gain(
        [
            ConformalGainObservation(
                record_id=state.record_id,
                predicted_mean_gain=prediction.expected_net_gain,
                predicted_scale=math.sqrt(prediction.total_variance),
                observed_gain=state.actual_net_gain,
            )
            for state, prediction in zip(states["feedback"], predictions["feedback"])
        ],
        alpha=0.10,
    )
    test_rows = []
    for state, prediction in zip(states["test"], predictions["test"]):
        lower = conformal.predict_one(
            record_id=state.record_id,
            predicted_mean_gain=prediction.expected_net_gain,
            predicted_scale=math.sqrt(prediction.total_variance),
        )
        test_rows.append(
            {
                **prediction.to_json(),
                "actual_net_gain": state.actual_net_gain,
                "lower_gain_bound": lower.lower_gain_bound,
            }
        )
    certificates = {
        f"alpha_{alpha:.2f}": _certify_threshold_family(
            states["acceptance"], predictions["acceptance"], alpha=alpha
        )
        for alpha in (0.10, 0.20)
    }
    deployments = {
        name: _deploy_certificate(certificate, states["test"], predictions["test"])
        for name, certificate in certificates.items()
    }
    labels = [int(state.actual_net_gain > 0.0) for state in states["test"]]
    direct_probabilities = [prediction.direct_worthiness for prediction in predictions["test"]]
    probabilities = [prediction.action_worthiness for prediction in predictions["test"]]
    return {
        "fold": fold_dir.name,
        "development_only_after_prior_test_inspection": True,
        "model": model.to_json(),
        "source_fuser": fuser.to_json(),
        "split_counts": {split: len(rows) for split, rows in states.items()},
        "metrics": {
            "direct_llm": binary_metrics(labels, direct_probabilities, threshold=0.5),
            "safe_evidence_residual": binary_metrics(labels, probabilities, threshold=0.5),
            "expected_gain_mae": _mean(
                abs(prediction.expected_net_gain - state.actual_net_gain)
                for state, prediction in zip(states["test"], predictions["test"])
            ),
            "expected_gain_correlation": _correlation(
                [prediction.expected_net_gain for prediction in predictions["test"]],
                [state.actual_net_gain for state in states["test"]],
            ),
            "test_conformal_coverage": _mean(
                float(row["actual_net_gain"] >= row["lower_gain_bound"])
                for row in test_rows
            ),
            "internal_variance_error_correlation": _variance_error_correlation(
                states["test"], predictions["test"], "internal_variance"
            ),
            "external_variance_error_correlation": _variance_error_correlation(
                states["test"], predictions["test"], "external_variance"
            ),
            "interaction_variance_error_correlation": _variance_error_correlation(
                states["test"], predictions["test"], "interaction_variance"
            ),
        },
        "conformal": conformal.to_json(),
        "certificates": certificates,
        "deployments": deployments,
        "test_rows": test_rows,
    }


def _candidate_excluded_train_states(
    records: list[ActionRecord], *, replicas: int
):
    output = []
    for record in records:
        excluded = set(record.metadata["candidate_group_ids"])
        reduced = [
            candidate
            for candidate in records
            if candidate.benchmark == record.benchmark
            and excluded.isdisjoint(candidate.metadata["candidate_group_ids"])
        ]
        model = CrossedPairwiseGainModel(
            replicas=replicas,
            neighbors=min(3, len(reduced)),
        ).fit(reduced)
        output.append(model.predict_one(record))
    return output


def _certify_threshold_family(states, predictions, *, alpha: float, delta: float = 0.05):
    candidates = []
    adjusted_delta = delta / len(THRESHOLDS)
    for threshold in THRESHOLDS:
        selected = [
            (state, prediction)
            for state, prediction in zip(states, predictions)
            if prediction.action_worthiness >= threshold
        ]
        errors = sum(state.actual_net_gain <= 0.0 for state, _ in selected)
        upper = binomial_upper_confidence(errors, len(selected), adjusted_delta)
        candidates.append(
            {
                "threshold": threshold,
                "selected_count": len(selected),
                "errors": errors,
                "empirical_risk": errors / len(selected) if selected else 0.0,
                "risk_upper_bound": upper,
                "certified": bool(selected) and upper <= alpha,
            }
        )
    certified = [row for row in candidates if row["certified"]]
    selected_policy = min(certified, key=lambda row: row["threshold"]) if certified else None
    return {
        "alpha": alpha,
        "delta": delta,
        "familywise_method": "Bonferroni exact Clopper-Pearson",
        "threshold_family": list(THRESHOLDS),
        "candidates": candidates,
        "selected_policy": selected_policy,
        "certified": selected_policy is not None,
    }


def _deploy_certificate(certificate, states, predictions):
    policy = certificate["selected_policy"]
    if policy is None:
        selected = []
    else:
        selected = [
            (state, prediction)
            for state, prediction in zip(states, predictions)
            if prediction.action_worthiness >= policy["threshold"]
        ]
    return {
        "selected_count": len(selected),
        "coverage": len(selected) / len(states) if states else 0.0,
        "selective_risk": _mean(
            float(state.actual_net_gain <= 0.0) for state, _ in selected
        ),
        "population_net_gain": (
            sum(state.actual_net_gain for state, _ in selected) / len(states)
            if states
            else 0.0
        ),
        "selected_mean_net_gain": _mean(
            state.actual_net_gain for state, _ in selected
        ),
    }


def _aggregate(reports):
    direct = [report["metrics"]["direct_llm"] for report in reports]
    ours = [report["metrics"]["safe_evidence_residual"] for report in reports]
    summary = {
        "folds": len(reports),
        "direct_llm": _mean_metrics(direct),
        "safe_evidence_residual": _mean_metrics(ours),
        "expected_gain_mae": _mean(
            report["metrics"]["expected_gain_mae"] for report in reports
        ),
        "expected_gain_correlation": _mean(
            report["metrics"]["expected_gain_correlation"] for report in reports
        ),
        "test_conformal_coverage": _mean(
            report["metrics"]["test_conformal_coverage"] for report in reports
        ),
        "certified_folds_alpha_0.10": sum(
            report["certificates"]["alpha_0.10"]["certified"] for report in reports
        ),
        "certified_folds_alpha_0.20": sum(
            report["certificates"]["alpha_0.20"]["certified"] for report in reports
        ),
        "population_net_gain_alpha_0.10": _mean(
            report["deployments"]["alpha_0.10"]["population_net_gain"]
            for report in reports
        ),
        "population_net_gain_alpha_0.20": _mean(
            report["deployments"]["alpha_0.20"]["population_net_gain"]
            for report in reports
        ),
    }
    return {"summary": summary, "fold_reports": reports}


def _mean_metrics(rows):
    return {key: _mean(row[key] for row in rows) for key in rows[0]}


def _variance_error_correlation(states, predictions, field):
    return _correlation(
        [getattr(prediction, field) for prediction in predictions],
        [
            (prediction.expected_net_gain - state.actual_net_gain) ** 2
            for state, prediction in zip(states, predictions)
        ],
    )


def _correlation(left, right):
    if len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _load(path: Path) -> list[ActionRecord]:
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
