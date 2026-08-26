from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from harness_matsci.adaptive_evidence_dataset import AdaptiveEvidenceExample, STAGES
from harness_matsci.adaptive_factorized_model import (
    CrossedFactorizedPrediction,
    fit_adaptive_factorized_gain_model,
)
from harness_matsci.adaptive_gain_harness import (
    counterfactual_evidence_value_cells,
    crossed_numeric_estimate,
    leave_one_out_topk_shadow_prices,
    sunk_cost_neutral_score_cells,
)
from harness_matsci.conformal_gain import (
    ConformalGainObservation,
    fit_split_conformal_gain,
)
from harness_matsci.learn_then_test_gain import (
    FixedThresholdGainPolicy,
    fit_learn_then_test_gain_policy,
)
from harness_matsci.portfolio_knowledge_gradient import (
    PortfolioKnowledgeGradientRow,
    deterministic_kg_partition,
    fit_portfolio_knowledge_gradient_model,
    portfolio_kg_features,
    realized_single_acquisition_portfolio_gain,
)
from harness_matsci.risk_control import binomial_upper_confidence
from harness_matsci.selective_execution_policy import (
    fit_task_balanced_execution_policy,
)


COST_WEIGHT = 0.15
DEFAULT_ALPHA = 0.10
DEFAULT_BUDGET = 0.10


@dataclass(frozen=True)
class StageEstimate:
    prediction: CrossedFactorizedPrediction
    gain_lower_bound: float = 0.0
    calibrated_worthiness: float = 0.5


@dataclass(frozen=True)
class EvidenceEstimate:
    mean: float
    predictive_variance: float
    internal_variance: float
    external_variance: float
    interaction_variance: float
    positive_probability: float
    lower_bound: float = 0.0
    current_gain_by_model: tuple[float, ...] = ()
    future_gain_cells: tuple[tuple[float, ...], ...] = ()
    decision_mean: float = 0.0
    decision_predictive_variance: float = 0.0
    decision_internal_variance: float = 0.0
    decision_external_variance: float = 0.0
    decision_interaction_variance: float = 0.0
    current_decision_score_by_model: tuple[float, ...] = ()
    future_decision_score_cells: tuple[tuple[float, ...], ...] = ()
    evidence_cost: float = 0.0
    current_cumulative_cost: float = 0.0


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float

    def predict(self, probability: float) -> float:
        logit = math.log(_clip_probability(probability) / (1.0 - _clip_probability(probability)))
        value = self.coefficient * logit + self.intercept
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--budget-fraction", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--development-evaluate-test", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()
    payload = run_experiment(
        Path(args.input),
        seed=args.seed,
        replicas=args.replicas,
        trees=args.trees,
        alpha=args.alpha,
        budget_fraction=args.budget_fraction,
        development_evaluate_test=args.development_evaluate_test,
        confirmatory=args.confirmatory,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def run_experiment(
    input_path: Path,
    *,
    seed: int,
    replicas: int,
    trees: int,
    alpha: float,
    budget_fraction: float,
    development_evaluate_test: bool,
    confirmatory: bool = False,
) -> dict[str, object]:
    if confirmatory and development_evaluate_test:
        raise ValueError("confirmatory acceptance cannot evaluate test")
    examples = [AdaptiveEvidenceExample(row) for row in _load_jsonl(input_path)]
    by_split = {
        split: [example for example in examples if example.split == split]
        for split in ("train", "feedback", "acceptance", "test")
    }
    if not by_split["train"] or not by_split["feedback"]:
        raise ValueError("H5 requires non-empty train and feedback splits")
    models = {
        stage: fit_adaptive_factorized_gain_model(
            [example.training_row(stage) for example in by_split["train"]],
            seed=seed + 1009 * stage_index,
            replicas=replicas,
            trees=trees,
        )
        for stage_index, stage in enumerate(STAGES)
    }
    evaluation_splits = ["feedback"]
    if by_split["acceptance"]:
        evaluation_splits.append("acceptance")
    if development_evaluate_test:
        if not by_split["test"]:
            raise ValueError("development test evaluation requires test examples")
        evaluation_splits.append("test")
    if confirmatory and not by_split["acceptance"]:
        raise ValueError("confirmatory evaluation requires acceptance examples")
    stage_estimates = {
        split: _predict_stages(by_split[split], models)
        for split in evaluation_splits
    }
    feedback_partitions = _feedback_partitions(by_split["feedback"], seed=seed)
    feedback_gain = [
        example
        for example in by_split["feedback"]
        if feedback_partitions[example.record_id] == "gain"
    ]
    feedback_evidence = [
        example
        for example in by_split["feedback"]
        if feedback_partitions[example.record_id] == "evidence"
    ]
    feedback_policy = [
        example
        for example in by_split["feedback"]
        if feedback_partitions[example.record_id] == "policy"
    ]
    gain_calibrators, platt_calibrators = _fit_gain_calibrators(
        feedback_gain,
        stage_estimates["feedback"],
        alpha=alpha,
    )
    stage_estimates = {
        split: _apply_gain_calibration(
            by_split[split], estimates, gain_calibrators, platt_calibrators
        )
        for split, estimates in stage_estimates.items()
    }
    evidence_estimates = {
        split: _predict_evidence_values(
            by_split[split],
            models,
            stage_estimates[split],
            gain_calibrators,
        )
        for split in evaluation_splits
    }
    portfolio_kg_model, portfolio_kg_calibrators, portfolio_kg_audit = (
        _fit_portfolio_kg_head(
            feedback_evidence,
            stage_estimates["feedback"],
            evidence_estimates["feedback"],
            alpha=alpha,
            budget_fraction=budget_fraction,
            seed=seed,
            task_balanced=False,
        )
    )
    task_portfolio_kg_model, task_portfolio_kg_calibrators, task_portfolio_kg_audit = (
        _fit_portfolio_kg_head(
            feedback_evidence,
            stage_estimates["feedback"],
            evidence_estimates["feedback"],
            alpha=alpha,
            budget_fraction=budget_fraction,
            seed=seed + 271,
            task_balanced=True,
        )
    )
    evidence_calibrators = _fit_evidence_calibrators(
        feedback_evidence,
        stage_estimates["feedback"],
        evidence_estimates["feedback"],
        alpha=alpha,
    )
    evidence_estimates = {
        split: _apply_evidence_calibration(
            by_split[split], estimates, evidence_calibrators
        )
        for split, estimates in evidence_estimates.items()
    }
    primary_policy, primary_policy_evidence = _fit_primary_policy(
        feedback_policy,
        stage_estimates["feedback"],
        evidence_estimates["feedback"],
        alpha=alpha,
        budget_fraction=budget_fraction,
        seed=seed,
    )
    learned_execution_policy, learned_execution_policy_evidence = (
        _fit_learned_execution_policy(
            feedback_policy,
            stage_estimates["feedback"],
            evidence_estimates["feedback"],
            task_portfolio_kg_model,
            task_portfolio_kg_calibrators,
            alpha=alpha,
            budget_fraction=budget_fraction,
        )
    )
    summary = {
        "feedback_partition": {
            "gain_calibration": len(feedback_gain),
            "evidence_calibration": len(feedback_evidence),
            "policy_ltt": len(feedback_policy),
        },
        "policy_feedback": _evaluate_methods(
            feedback_policy,
            stage_estimates["feedback"],
            evidence_estimates["feedback"],
            budget_fraction=budget_fraction,
            primary_policy=primary_policy,
            portfolio_kg_model=portfolio_kg_model,
            portfolio_kg_calibrators=portfolio_kg_calibrators,
            task_portfolio_kg_model=task_portfolio_kg_model,
            task_portfolio_kg_calibrators=task_portfolio_kg_calibrators,
            learned_execution_policy=learned_execution_policy,
            alpha=alpha,
        ),
        "acceptance": (
            _evaluate_methods(
                by_split["acceptance"],
                stage_estimates["acceptance"],
                evidence_estimates["acceptance"],
                budget_fraction=budget_fraction,
                primary_policy=primary_policy,
                portfolio_kg_model=portfolio_kg_model,
                portfolio_kg_calibrators=portfolio_kg_calibrators,
                task_portfolio_kg_model=task_portfolio_kg_model,
                task_portfolio_kg_calibrators=task_portfolio_kg_calibrators,
                learned_execution_policy=learned_execution_policy,
                alpha=alpha,
            )
            if by_split["acceptance"]
            else None
        ),
        "test": (
            _evaluate_methods(
                by_split["test"],
                stage_estimates["test"],
                evidence_estimates["test"],
                budget_fraction=budget_fraction,
                primary_policy=primary_policy,
                portfolio_kg_model=portfolio_kg_model,
                portfolio_kg_calibrators=portfolio_kg_calibrators,
                task_portfolio_kg_model=task_portfolio_kg_model,
                task_portfolio_kg_calibrators=task_portfolio_kg_calibrators,
                learned_execution_policy=learned_execution_policy,
                alpha=alpha,
            )
            if development_evaluate_test
            else None
        ),
        "calibration": {
            split: _calibration_metrics(by_split[split], stage_estimates[split])
            for split in evaluation_splits
        },
        "portfolio_kg_audit": portfolio_kg_audit,
        "task_balanced_portfolio_kg_audit": task_portfolio_kg_audit,
    }
    return {
        "schema_version": "h5-cost-fair-adaptive-evidence-v2",
        "method": {
            "output": "numeric action-worthiness and numeric evidence value only",
            "strict_gain": "S*U - 0.15*(action_cost + acquired_evidence_cost) - 0.25*(1-S)*failure_harm",
            "evidence_value": "decision-consistent crossed portfolio knowledge gain over a leave-one-out top-k shadow price, with conformal execution scores, sunk-cost neutrality, and incremental evidence cost charged",
            "uncertainty": "balanced model-replica x evidence-replica ANOVA plus factorized aleatoric variance",
            "calibration": "task-stage Platt action-worthiness plus one-sided normalized split conformal gain and evidence-value bounds",
            "route_output": False,
        },
        "claim_boundary": {
            "development_only": not confirmatory,
            "confirmatory": bool(confirmatory),
            "test_inspected": bool(development_evaluate_test),
            "matbot_runtime": False,
            "observed_tools": "controlled noisy task-native measurements",
            "pre_acquisition_priors": "train-only structured surrogate replicas",
            "selective_risk_certificate": bool(confirmatory),
        },
        "input": str(input_path),
        "seed": seed,
        "replicas": replicas,
        "trees": trees,
        "alpha": alpha,
        "budget_fraction": budget_fraction,
        "sizes": {split: len(rows) for split, rows in by_split.items()},
        "models": {stage: model.to_json() for stage, model in models.items()},
        "gain_calibrators": {
            key: calibrator.to_json() for key, calibrator in sorted(gain_calibrators.items())
        },
        "evidence_calibrators": {
            key: calibrator.to_json()
            for key, calibrator in sorted(evidence_calibrators.items())
        },
        "platt_calibrators": {
            key: {
                "coefficient": calibrator.coefficient,
                "intercept": calibrator.intercept,
            }
            for key, calibrator in sorted(platt_calibrators.items())
        },
        "primary_ltt_policy": (
            primary_policy.to_json() if primary_policy is not None else None
        ),
        "primary_ltt_candidates": [
            row.to_json() for row in primary_policy_evidence
        ],
        "portfolio_kg_model": portfolio_kg_model.to_json(),
        "portfolio_kg_calibrators": {
            key: calibrator.to_json()
            for key, calibrator in sorted(portfolio_kg_calibrators.items())
        },
        "task_balanced_portfolio_kg_model": task_portfolio_kg_model.to_json(),
        "task_balanced_portfolio_kg_calibrators": {
            key: calibrator.to_json()
            for key, calibrator in sorted(task_portfolio_kg_calibrators.items())
        },
        "learned_execution_policy": (
            learned_execution_policy.to_json()
            if learned_execution_policy is not None
            else None
        ),
        "learned_execution_policy_candidates": [
            row.to_json() for row in learned_execution_policy_evidence
        ],
        "summary": summary,
    }


def _predict_stages(examples, models):
    output = {}
    for example in examples:
        output[example.record_id] = {}
        for stage in STAGES:
            prediction = models[stage].predict_crossed(
                record_id=example.record_id,
                benchmark=example.benchmark,
                feature_views=example.current_feature_views(stage),
                normalized_action_cost=example.normalized_action_cost,
                failure_harm=example.failure_harm,
                cumulative_evidence_cost=example.cumulative_evidence_cost(stage),
            )
            output[example.record_id][stage] = StageEstimate(prediction)
    return output


def _predict_evidence_values(
    examples,
    models,
    stage_estimates,
    gain_calibrators,
):
    output = {}
    for example in examples:
        output[example.record_id] = {}
        for current_stage, next_stage in (("base", "screening"), ("screening", "verification")):
            current = stage_estimates[example.record_id][current_stage].prediction
            future = models[next_stage].predict_crossed(
                record_id=example.record_id,
                benchmark=example.benchmark,
                feature_views=example.counterfactual_next_stage_views(current_stage),
                normalized_action_cost=example.normalized_action_cost,
                failure_harm=example.failure_harm,
                cumulative_evidence_cost=example.cumulative_evidence_cost(current_stage),
            )
            current_by_model = [sum(row) / len(row) for row in current.gain_cells]
            current_cost = example.cumulative_evidence_cost(current_stage)
            cells = counterfactual_evidence_value_cells(
                current_by_model,
                future.gain_cells,
                evidence_cost=example.next_evidence_cost(current_stage),
                cost_weight=COST_WEIGHT,
                current_abstention_value=-COST_WEIGHT * current_cost,
                future_abstention_value=-COST_WEIGHT * current_cost,
            )
            estimate = crossed_numeric_estimate(
                cells,
                aleatoric_variance=(
                    current.aleatoric_gain_variance + future.aleatoric_gain_variance
                ),
            )
            current_radius = _gain_conformal_radius(
                gain_calibrators[f"{example.benchmark}::{current_stage}"],
                current.predictive_gain_variance,
            )
            future_radius = _gain_conformal_radius(
                gain_calibrators[f"{example.benchmark}::{next_stage}"],
                future.predictive_gain_variance,
            )
            current_score_cells = sunk_cost_neutral_score_cells(
                current.gain_cells,
                cumulative_evidence_cost=current_cost,
                cost_weight=COST_WEIGHT,
                lower_bound_radius=current_radius,
            )
            future_score_cells = sunk_cost_neutral_score_cells(
                future.gain_cells,
                cumulative_evidence_cost=current_cost,
                cost_weight=COST_WEIGHT,
                lower_bound_radius=future_radius,
            )
            current_decision_by_model = [
                sum(row) / len(row) for row in current_score_cells
            ]
            decision_cells = counterfactual_evidence_value_cells(
                current_decision_by_model,
                future_score_cells,
                evidence_cost=example.next_evidence_cost(current_stage),
                cost_weight=COST_WEIGHT,
            )
            decision_estimate = crossed_numeric_estimate(decision_cells)
            output[example.record_id][next_stage] = EvidenceEstimate(
                mean=estimate.mean,
                predictive_variance=estimate.predictive_variance,
                internal_variance=estimate.internal_variance,
                external_variance=estimate.external_variance,
                interaction_variance=estimate.interaction_variance,
                positive_probability=estimate.positive_probability,
                current_gain_by_model=tuple(current_by_model),
                future_gain_cells=tuple(tuple(row) for row in future.gain_cells),
                decision_mean=decision_estimate.mean,
                decision_predictive_variance=decision_estimate.predictive_variance,
                decision_internal_variance=decision_estimate.internal_variance,
                decision_external_variance=decision_estimate.external_variance,
                decision_interaction_variance=decision_estimate.interaction_variance,
                current_decision_score_by_model=tuple(current_decision_by_model),
                future_decision_score_cells=tuple(
                    tuple(row) for row in future_score_cells
                ),
                evidence_cost=example.next_evidence_cost(current_stage),
                current_cumulative_cost=current_cost,
            )
    return output


def _gain_conformal_radius(calibrator, predictive_variance):
    scale = max(
        calibrator.minimum_scale,
        math.sqrt(max(1e-10, float(predictive_variance))),
    )
    return max(0.0, calibrator.normalized_radius * scale)


def _fit_portfolio_kg_head(
    examples,
    stage_estimates,
    evidence_estimates,
    *,
    alpha,
    budget_fraction,
    seed,
    task_balanced,
):
    all_rows = _portfolio_kg_rows(
        examples,
        stage_estimates,
        evidence_estimates,
        budget_fraction=budget_fraction,
        task_balanced=task_balanced,
    )
    fit_rows = [
        row
        for row in all_rows
        if deterministic_kg_partition(
            row.record_id,
            row.benchmark,
            seed=seed,
        )
        == "fit"
    ]
    calibration_rows = [
        row
        for row in all_rows
        if deterministic_kg_partition(
            row.record_id,
            row.benchmark,
            seed=seed,
        )
        == "calibrate"
    ]
    model = fit_portfolio_knowledge_gradient_model(
        fit_rows,
        seed=seed + 15401,
        replicas=3,
        trees=80,
        min_samples_leaf=5,
    )
    calibrators = {}
    audit = {
        "portfolio_context_examples": len(examples),
        "task_balanced": bool(task_balanced),
        "fit_examples": len({row.record_id for row in fit_rows}),
        "calibration_examples": len(
            {row.record_id for row in calibration_rows}
        ),
        "fit_heads": _portfolio_kg_target_audit(fit_rows),
        "heads": {},
    }
    heads = sorted(
        {(row.benchmark, row.transition) for row in calibration_rows}
    )
    for benchmark, transition in heads:
        head_rows = [
            row
            for row in calibration_rows
            if row.benchmark == benchmark and row.transition == transition
        ]
        observations = []
        absolute_errors = []
        for row in head_rows:
            prediction = model.predict(
                record_id=row.record_id,
                benchmark=row.benchmark,
                transition=row.transition,
                features=row.features,
            )
            observations.append(
                ConformalGainObservation(
                    record_id=row.record_id,
                    predicted_mean_gain=prediction.expected_incremental_gain,
                    predicted_scale=math.sqrt(
                        max(1e-10, prediction.predictive_variance)
                    ),
                    observed_gain=row.realized_incremental_gain,
                )
            )
            absolute_errors.append(
                abs(
                    prediction.expected_incremental_gain
                    - row.realized_incremental_gain
                )
            )
        key = f"{benchmark}::{transition}"
        calibrators[key] = fit_split_conformal_gain(observations, alpha=alpha)
        audit["heads"][key] = {
            "count": len(head_rows),
            "positive_rate": sum(
                row.realized_incremental_gain > 0.0 for row in head_rows
            )
            / len(head_rows),
            "mean_realized_incremental_gain": sum(
                row.realized_incremental_gain for row in head_rows
            )
            / len(head_rows),
            "mean_absolute_error": sum(absolute_errors) / len(absolute_errors),
        }
    return model, calibrators, audit


def _portfolio_kg_target_audit(rows):
    output = {}
    heads = sorted({(row.benchmark, row.transition) for row in rows})
    for benchmark, transition in heads:
        head_rows = [
            row
            for row in rows
            if row.benchmark == benchmark and row.transition == transition
        ]
        output[f"{benchmark}::{transition}"] = {
            "count": len(head_rows),
            "positive_rate": sum(
                row.realized_incremental_gain > 0.0 for row in head_rows
            )
            / len(head_rows),
            "mean_realized_incremental_gain": sum(
                row.realized_incremental_gain for row in head_rows
            )
            / len(head_rows),
        }
    return output


def _portfolio_kg_rows(
    examples,
    stage_estimates,
    evidence_estimates,
    *,
    budget_fraction,
    task_balanced,
):
    rows = []
    for current_stage, next_stage in (
        ("base", "screening"),
        ("screening", "verification"),
    ):
        transition = f"{current_stage}->{next_stage}"
        groups = (
            [
                [
                    example
                    for example in examples
                    if example.benchmark == benchmark
                ]
                for benchmark in sorted({example.benchmark for example in examples})
            ]
            if task_balanced
            else [list(examples)]
        )
        for task_examples in groups:
            stages = {
                example.record_id: current_stage for example in task_examples
            }
            current_scores = [
                _decision_score(
                    example,
                    current_stage,
                    stage_estimates[example.record_id][current_stage],
                )
                for example in task_examples
            ]
            current_action_gains = [
                example.realized_gain(current_stage)
                + COST_WEIGHT * example.cumulative_evidence_cost(current_stage)
                for example in task_examples
            ]
            shadows = _leave_one_out_execution_shadows(
                task_examples,
                stages,
                stage_estimates,
                budget_fraction,
            )
            budget = max(1, math.ceil(budget_fraction * len(task_examples)))
            for index, example in enumerate(task_examples):
                estimate = evidence_estimates[example.record_id][next_stage]
                structural = _decision_consistent_crossed_estimate(
                    estimate,
                    shadow_price=shadows[index],
                )
                features = portfolio_kg_features(
                    current_score=current_scores[index],
                    shadow_price=shadows[index],
                    structural_mean=structural.mean,
                    structural_variance=structural.predictive_variance,
                    internal_variance=structural.internal_variance,
                    external_variance=structural.external_variance,
                    interaction_variance=structural.interaction_variance,
                    positive_probability=structural.positive_probability,
                    evidence_cost=estimate.evidence_cost,
                    future_score_cells=estimate.future_decision_score_cells,
                )
                future_action_gain = (
                    example.realized_gain(next_stage)
                    + COST_WEIGHT * example.cumulative_evidence_cost(next_stage)
                )
                target = realized_single_acquisition_portfolio_gain(
                    current_scores=current_scores,
                    current_action_gains=current_action_gains,
                    candidate_index=index,
                    future_score=_decision_score(
                        example,
                        next_stage,
                        stage_estimates[example.record_id][next_stage],
                    ),
                    future_action_gain=future_action_gain,
                    budget=budget,
                    charged_evidence_cost=(
                        COST_WEIGHT * example.next_evidence_cost(current_stage)
                    ),
                )
                rows.append(
                    PortfolioKnowledgeGradientRow(
                        record_id=example.record_id,
                        benchmark=example.benchmark,
                        transition=transition,
                        features=features,
                        realized_incremental_gain=target,
                    )
                )
    return rows


def _fit_gain_calibrators(examples, estimates, *, alpha):
    from sklearn.linear_model import LogisticRegression

    conformal = {}
    platt = {}
    for benchmark in sorted({example.benchmark for example in examples}):
        task_rows = [example for example in examples if example.benchmark == benchmark]
        for stage in STAGES:
            key = f"{benchmark}::{stage}"
            observations = []
            logits = []
            labels = []
            for example in task_rows:
                prediction = estimates[example.record_id][stage].prediction
                observations.append(
                    ConformalGainObservation(
                        record_id=example.record_id,
                        predicted_mean_gain=prediction.expected_net_gain,
                        predicted_scale=math.sqrt(max(1e-10, prediction.predictive_gain_variance)),
                        observed_gain=example.realized_gain(stage),
                    )
                )
                probability = _clip_probability(prediction.p_positive_gain)
                logits.append([math.log(probability / (1.0 - probability))])
                labels.append(int(example.realized_gain(stage) > 0.0))
            conformal[key] = fit_split_conformal_gain(observations, alpha=alpha)
            if len(set(labels)) < 2:
                platt[key] = PlattCalibrator(1.0, 0.0)
            else:
                model = LogisticRegression(C=1.0, max_iter=2000)
                model.fit(logits, labels)
                platt[key] = PlattCalibrator(
                    float(model.coef_[0][0]), float(model.intercept_[0])
                )
    return conformal, platt


def _apply_gain_calibration(examples, estimates, conformal, platt):
    output = {}
    for example in examples:
        output[example.record_id] = {}
        for stage in STAGES:
            raw = estimates[example.record_id][stage]
            key = f"{example.benchmark}::{stage}"
            prediction = raw.prediction
            bound = conformal[key].predict_one(
                record_id=example.record_id,
                predicted_mean_gain=prediction.expected_net_gain,
                predicted_scale=math.sqrt(max(1e-10, prediction.predictive_gain_variance)),
            )
            output[example.record_id][stage] = StageEstimate(
                prediction=prediction,
                gain_lower_bound=bound.lower_gain_bound,
                calibrated_worthiness=platt[key].predict(prediction.p_positive_gain),
            )
    return output


def _fit_evidence_calibrators(examples, stage_estimates, evidence_estimates, *, alpha):
    calibrators = {}
    for benchmark in sorted({example.benchmark for example in examples}):
        task_rows = [example for example in examples if example.benchmark == benchmark]
        for current_stage, next_stage in (("base", "screening"), ("screening", "verification")):
            key = f"{benchmark}::{current_stage}->{next_stage}"
            observations = []
            for example in task_rows:
                estimate = evidence_estimates[example.record_id][next_stage]
                current = stage_estimates[example.record_id][current_stage]
                future = stage_estimates[example.record_id][next_stage]
                current_abstain = -COST_WEIGHT * example.cumulative_evidence_cost(current_stage)
                future_abstain = -COST_WEIGHT * example.cumulative_evidence_cost(next_stage)
                current_execute = current.gain_lower_bound > current_abstain
                future_execute = future.gain_lower_bound > future_abstain
                before = (
                    example.realized_gain(current_stage)
                    if current_execute
                    else current_abstain
                )
                after = (
                    example.realized_gain(next_stage)
                    if future_execute
                    else future_abstain
                )
                observations.append(
                    ConformalGainObservation(
                        record_id=example.record_id,
                        predicted_mean_gain=estimate.mean,
                        predicted_scale=math.sqrt(max(1e-10, estimate.predictive_variance)),
                        observed_gain=after - before,
                    )
                )
            calibrators[key] = fit_split_conformal_gain(observations, alpha=alpha)
    return calibrators


def _apply_evidence_calibration(examples, estimates, calibrators):
    output = {}
    benchmark_by_id = {example.record_id: example.benchmark for example in examples}
    for record_id, by_stage in estimates.items():
        output[record_id] = {}
        for next_stage, estimate in by_stage.items():
            current_stage = "base" if next_stage == "screening" else "screening"
            key = (
                f"{benchmark_by_id[record_id]}::"
                f"{current_stage}->{next_stage}"
            )
            bound = calibrators[key].predict_one(
                record_id=record_id,
                predicted_mean_gain=estimate.mean,
                predicted_scale=math.sqrt(max(1e-10, estimate.predictive_variance)),
            )
            output[record_id][next_stage] = EvidenceEstimate(
                mean=estimate.mean,
                predictive_variance=estimate.predictive_variance,
                internal_variance=estimate.internal_variance,
                external_variance=estimate.external_variance,
                interaction_variance=estimate.interaction_variance,
                positive_probability=estimate.positive_probability,
                lower_bound=bound.lower_gain_bound,
                current_gain_by_model=estimate.current_gain_by_model,
                future_gain_cells=estimate.future_gain_cells,
                decision_mean=estimate.decision_mean,
                decision_predictive_variance=estimate.decision_predictive_variance,
                decision_internal_variance=estimate.decision_internal_variance,
                decision_external_variance=estimate.decision_external_variance,
                decision_interaction_variance=estimate.decision_interaction_variance,
                current_decision_score_by_model=(
                    estimate.current_decision_score_by_model
                ),
                future_decision_score_cells=estimate.future_decision_score_cells,
                evidence_cost=estimate.evidence_cost,
                current_cumulative_cost=estimate.current_cumulative_cost,
            )
    return output


def _evaluate_methods(
    examples,
    stage_estimates,
    evidence_estimates,
    *,
    budget_fraction,
    primary_policy,
    portfolio_kg_model,
    portfolio_kg_calibrators,
    task_portfolio_kg_model,
    task_portfolio_kg_calibrators,
    learned_execution_policy,
    alpha,
):
    methods = {
        "base_factorized_lcb": {example.record_id: "base" for example in examples},
        "fixed_screening_lcb": {example.record_id: "screening" for example in examples},
        "fixed_full_evidence_lcb": {example.record_id: "verification" for example in examples},
        "adaptive_mean_voi": {
            example.record_id: _adaptive_stage(example, evidence_estimates, conservative=False)
            for example in examples
        },
        "adaptive_conservative_voi": {
            example.record_id: _adaptive_stage(example, evidence_estimates, conservative=True)
            for example in examples
        },
        "adaptive_ambiguity_voi": {
            example.record_id: _adaptive_ambiguity_stage(
                example, stage_estimates, evidence_estimates
            )
            for example in examples
        },
        "static_reliability": {example.record_id: "base" for example in examples},
    }
    methods["shadow_price_adaptive_mean"] = _shadow_price_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        uncertainty_radius=0.0,
    )
    methods["shadow_price_adaptive_conservative"] = _shadow_price_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        uncertainty_radius=1.0,
    )
    methods["shadow_price_successive_halving"] = _shadow_price_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        uncertainty_radius=1.0,
        fill_screening_budget=True,
        fill_verification_budget=True,
    )
    methods["decision_consistent_kg_mean"] = _decision_consistent_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        uncertainty_radius=0.0,
    )
    methods["decision_consistent_safe_kg"] = _decision_consistent_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        uncertainty_radius=1.0,
    )
    methods["learned_portfolio_kg_mean"] = _learned_portfolio_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        portfolio_kg_model,
        portfolio_kg_calibrators,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        use_conformal_bound=False,
        task_balanced=False,
    )
    methods["learned_portfolio_kg_lcb"] = _learned_portfolio_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        portfolio_kg_model,
        portfolio_kg_calibrators,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        use_conformal_bound=True,
        task_balanced=False,
    )
    methods["learned_portfolio_kg_mean_task_balanced"] = _learned_portfolio_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        task_portfolio_kg_model,
        task_portfolio_kg_calibrators,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        use_conformal_bound=False,
        task_balanced=True,
    )
    methods["learned_portfolio_kg_lcb_task_balanced"] = _learned_portfolio_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        task_portfolio_kg_model,
        task_portfolio_kg_calibrators,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        use_conformal_bound=True,
        task_balanced=True,
    )
    methods["base_factorized_lcb_task_balanced"] = methods[
        "base_factorized_lcb"
    ]
    methods["fixed_screening_lcb_task_balanced"] = methods[
        "fixed_screening_lcb"
    ]
    methods["fixed_full_evidence_lcb_task_balanced"] = methods[
        "fixed_full_evidence_lcb"
    ]
    task_balanced_methods = {
        "learned_portfolio_kg_mean_task_balanced",
        "learned_portfolio_kg_lcb_task_balanced",
        "base_factorized_lcb_task_balanced",
        "fixed_screening_lcb_task_balanced",
        "fixed_full_evidence_lcb_task_balanced",
    }
    output = {}
    for method, stages in methods.items():
        if method == "static_reliability":
            scores = [_static_reliability(example) for example in examples]
        else:
            scores = [
                _decision_score(
                    example,
                    stages[example.record_id],
                    stage_estimates[example.record_id][stages[example.record_id]],
                )
                for example in examples
            ]
        output[method] = (
            _evaluate_task_balanced_trajectory_policy(
                examples,
                stages,
                scores,
                budget_fraction=budget_fraction,
            )
            if method in task_balanced_methods
            else _evaluate_trajectory_policy(
                examples,
                stages,
                scores,
                budget_fraction=budget_fraction,
            )
        )
        output[method] = _attach_batch_certificate(output[method], alpha=alpha)
    conservative_voi_stages = methods["adaptive_conservative_voi"]
    calibrated_scores = [
        stage_estimates[example.record_id][
            conservative_voi_stages[example.record_id]
        ].calibrated_worthiness
        for example in examples
    ]
    output["adaptive_conservative_calibrated_p"] = _evaluate_trajectory_policy(
        examples,
        conservative_voi_stages,
        calibrated_scores,
        budget_fraction=budget_fraction,
    )
    output["adaptive_conservative_calibrated_p"] = _attach_batch_certificate(
        output["adaptive_conservative_calibrated_p"], alpha=alpha
    )
    conservative_stages = methods["adaptive_ambiguity_voi"]
    conservative_scores = [
        stage_estimates[example.record_id][
            conservative_stages[example.record_id]
        ].gain_lower_bound
        - (
            -COST_WEIGHT
            * example.cumulative_evidence_cost(
                conservative_stages[example.record_id]
            )
        )
        for example in examples
    ]
    output["adaptive_conservative_ltt"] = _evaluate_fixed_policy(
        examples,
        conservative_stages,
        conservative_scores,
        primary_policy,
        alpha=alpha,
    )
    output["learned_portfolio_kg_risk_controlled"] = (
        _evaluate_learned_execution_policy(
            examples,
            stage_estimates,
            evidence_estimates,
            portfolio_kg_model,
            portfolio_kg_calibrators,
            learned_execution_policy,
            budget_fraction=budget_fraction,
            alpha=alpha,
        )
    )
    output["oracle_stage_and_execution"] = _oracle_evaluation(
        examples, budget_fraction=budget_fraction
    )
    output["oracle_stage_and_execution"] = _attach_batch_certificate(
        output["oracle_stage_and_execution"], alpha=alpha
    )
    return output


def _attach_batch_certificate(result, *, alpha):
    count = int(result["selected_count"])
    errors = int(result["errors"])
    upper = binomial_upper_confidence(errors, count, 0.05) if count else 1.0
    return {
        **result,
        "risk_upper_bound": upper,
        "alpha": alpha,
        "certified": bool(count and upper <= alpha),
        "certificate_scope": (
            "independent held-out batch certificate; batch ranking couples actions "
            "and is not an assumption-free action-wise theorem"
        ),
    }


def _fit_primary_policy(
    examples,
    stage_estimates,
    evidence_estimates,
    *,
    alpha,
    budget_fraction,
    seed,
):
    stages = {
        example.record_id: _adaptive_ambiguity_stage(
            example, stage_estimates, evidence_estimates
        )
        for example in examples
    }
    scores = [
        stage_estimates[example.record_id][
            stages[example.record_id]
        ].calibrated_worthiness
        for example in examples
    ]
    gains = [example.realized_gain(stages[example.record_id]) for example in examples]
    thresholds = (0.50, 0.70, 0.80, 0.90, 0.95)
    fitted, evidence = fit_learn_then_test_gain_policy(
        candidate_thresholds=thresholds,
        feedback_scores=scores,
        feedback_gains=gains,
        feedback_record_ids=[example.record_id for example in examples],
        alpha=alpha,
        delta=0.05,
        budget_fraction=1.0,
        thinning_seed=seed + 7919,
    )
    if fitted is None:
        return None, evidence
    return (
        FixedThresholdGainPolicy(
            threshold=fitted.threshold,
            budget_fraction=budget_fraction,
            thinning_seed=fitted.thinning_seed,
        ),
        evidence,
    )


def _fit_learned_execution_policy(
    examples,
    stage_estimates,
    evidence_estimates,
    portfolio_kg_model,
    portfolio_kg_calibrators,
    *,
    alpha,
    budget_fraction,
):
    stages = _learned_portfolio_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        portfolio_kg_model,
        portfolio_kg_calibrators,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        use_conformal_bound=False,
        task_balanced=True,
    )
    decision_scores = [
        _decision_score(
            example,
            stages[example.record_id],
            stage_estimates[example.record_id][stages[example.record_id]],
        )
        for example in examples
    ]
    worthiness = [
        stage_estimates[example.record_id][
            stages[example.record_id]
        ].calibrated_worthiness
        for example in examples
    ]
    realized = [
        example.realized_gain(stages[example.record_id]) for example in examples
    ]
    return fit_task_balanced_execution_policy(
        candidate_thresholds=(
            0.50,
            0.60,
            0.70,
            0.80,
            0.85,
            0.90,
            0.925,
            0.95,
            0.975,
            0.99,
        ),
        decision_scores=decision_scores,
        worthiness_probabilities=worthiness,
        realized_gains=realized,
        benchmarks=[example.benchmark for example in examples],
        record_ids=[example.record_id for example in examples],
        budget_fraction=budget_fraction,
        alpha=alpha,
        delta=0.05,
    )


def _evaluate_learned_execution_policy(
    examples,
    stage_estimates,
    evidence_estimates,
    portfolio_kg_model,
    portfolio_kg_calibrators,
    policy,
    *,
    budget_fraction,
    alpha,
):
    if policy is None:
        return {
            "available": False,
            "certified": False,
            "reason": "no simultaneously risk-certified worthiness threshold",
        }
    stages = _learned_portfolio_kg_stages(
        examples,
        stage_estimates,
        evidence_estimates,
        portfolio_kg_model,
        portfolio_kg_calibrators,
        execution_budget_fraction=budget_fraction,
        screening_budget_fraction=0.20,
        verification_budget_fraction=0.10,
        use_conformal_bound=False,
        task_balanced=True,
    )
    decision_scores = [
        _decision_score(
            example,
            stages[example.record_id],
            stage_estimates[example.record_id][stages[example.record_id]],
        )
        for example in examples
    ]
    worthiness = [
        stage_estimates[example.record_id][
            stages[example.record_id]
        ].calibrated_worthiness
        for example in examples
    ]
    selected = set(
        policy.select(
            decision_scores=decision_scores,
            worthiness_probabilities=worthiness,
            benchmarks=[example.benchmark for example in examples],
            record_ids=[example.record_id for example in examples],
        )
    )
    result = _evaluate_selected_indices(examples, stages, selected)
    return {
        "available": True,
        "policy": policy.to_json(),
        **_attach_batch_certificate(result, alpha=alpha),
        "feedback_guarantee": (
            "Bonferroni-simultaneous one-sided Clopper-Pearson testing over a "
            "predefined calibrated-worthiness threshold grid"
        ),
    }


def _evaluate_fixed_policy(examples, stages, scores, policy, *, alpha):
    if policy is None:
        return {
            "available": False,
            "certified": False,
            "reason": "no simultaneously certified LTT threshold on policy feedback",
        }
    selected = set(
        policy.select(scores, [example.record_id for example in examples])
    )
    result = _evaluate_selected_indices(examples, stages, selected)
    upper = (
        binomial_upper_confidence(result["errors"], result["selected_count"], 0.05)
        if result["selected_count"]
        else 1.0
    )
    return {
        "available": True,
        "policy": policy.to_json(),
        **result,
        "risk_upper_bound": upper,
        "alpha": alpha,
        "certified": bool(result["selected_count"] and upper <= alpha),
        "guarantee": (
            "one-sided Clopper-Pearson audit for a feedback-frozen action-wise "
            "threshold with outcome-independent budget thinning"
        ),
    }


def _adaptive_stage(example, evidence_estimates, *, conservative):
    values = evidence_estimates[example.record_id]
    screening_value = values["screening"].lower_bound if conservative else values["screening"].mean
    if screening_value <= 0.0:
        return "base"
    verification_value = values["verification"].lower_bound if conservative else values["verification"].mean
    return "verification" if verification_value > 0.0 else "screening"


def _adaptive_ambiguity_stage(example, stage_estimates, evidence_estimates):
    def should_acquire(current_stage, next_stage):
        current = stage_estimates[example.record_id][current_stage]
        abstention = -COST_WEIGHT * example.cumulative_evidence_cost(current_stage)
        reducible_ambiguity = (
            current.prediction.expected_net_gain > abstention
            and current.gain_lower_bound <= abstention
        )
        conservative_voi = evidence_estimates[example.record_id][next_stage].lower_bound
        return conservative_voi > 0.0 or reducible_ambiguity

    if not should_acquire("base", "screening"):
        return "base"
    return (
        "verification"
        if should_acquire("screening", "verification")
        else "screening"
    )


def _shadow_price_stages(
    examples,
    stage_estimates,
    evidence_estimates,
    *,
    execution_budget_fraction,
    screening_budget_fraction,
    verification_budget_fraction,
    uncertainty_radius,
    fill_screening_budget=False,
    fill_verification_budget=False,
):
    stages = {example.record_id: "base" for example in examples}
    screening_cutoff = _execution_shadow_price(
        examples,
        stages,
        stage_estimates,
        execution_budget_fraction,
    )
    screening_scores = []
    for index, example in enumerate(examples):
        estimate = _shadow_evidence_estimate(
            evidence_estimates[example.record_id]["screening"],
            shadow_price=screening_cutoff,
            uncertainty_radius=uncertainty_radius,
        )
        screening_scores.append((estimate, index))
    screening_budget = max(1, math.ceil(screening_budget_fraction * len(examples)))
    ranked_screening = sorted(
        screening_scores,
        key=lambda row: (-row[0], examples[row[1]].record_id),
    )[:screening_budget]
    screened = {
        index
        for score, index in ranked_screening
        if fill_screening_budget or score > 0.0
    }
    for index in screened:
        stages[examples[index].record_id] = "screening"
    verification_cutoff = _execution_shadow_price(
        examples,
        stages,
        stage_estimates,
        execution_budget_fraction,
    )
    verification_scores = []
    for index in screened:
        example = examples[index]
        estimate = _shadow_evidence_estimate(
            evidence_estimates[example.record_id]["verification"],
            shadow_price=verification_cutoff,
            uncertainty_radius=uncertainty_radius,
        )
        verification_scores.append((estimate, index))
    verification_budget = max(
        1, math.ceil(verification_budget_fraction * len(examples))
    )
    ranked_verification = sorted(
        verification_scores,
        key=lambda row: (-row[0], examples[row[1]].record_id),
    )[:verification_budget]
    verified = {
        index
        for score, index in ranked_verification
        if fill_verification_budget or score > 0.0
    }
    for index in verified:
        stages[examples[index].record_id] = "verification"
    return stages


def _decision_consistent_kg_stages(
    examples,
    stage_estimates,
    evidence_estimates,
    *,
    execution_budget_fraction,
    screening_budget_fraction,
    verification_budget_fraction,
    uncertainty_radius,
):
    stages = {example.record_id: "base" for example in examples}
    screening_shadows = _leave_one_out_execution_shadows(
        examples,
        stages,
        stage_estimates,
        execution_budget_fraction,
    )
    screening_scores = []
    for index, example in enumerate(examples):
        score = _decision_consistent_evidence_score(
            evidence_estimates[example.record_id]["screening"],
            shadow_price=screening_shadows[index],
            uncertainty_radius=uncertainty_radius,
        )
        screening_scores.append((score, index))
    screening_budget = max(1, math.ceil(screening_budget_fraction * len(examples)))
    screened = {
        index
        for score, index in sorted(
            screening_scores,
            key=lambda row: (-row[0], examples[row[1]].record_id),
        )[:screening_budget]
        if score > 0.0
    }
    for index in screened:
        stages[examples[index].record_id] = "screening"

    verification_shadows = _leave_one_out_execution_shadows(
        examples,
        stages,
        stage_estimates,
        execution_budget_fraction,
    )
    verification_scores = []
    for index in screened:
        example = examples[index]
        score = _decision_consistent_evidence_score(
            evidence_estimates[example.record_id]["verification"],
            shadow_price=verification_shadows[index],
            uncertainty_radius=uncertainty_radius,
        )
        verification_scores.append((score, index))
    verification_budget = max(
        1, math.ceil(verification_budget_fraction * len(examples))
    )
    verified = {
        index
        for score, index in sorted(
            verification_scores,
            key=lambda row: (-row[0], examples[row[1]].record_id),
        )[:verification_budget]
        if score > 0.0
    }
    for index in verified:
        stages[examples[index].record_id] = "verification"
    return stages


def _learned_portfolio_kg_stages(
    examples,
    stage_estimates,
    evidence_estimates,
    model,
    calibrators,
    *,
    execution_budget_fraction,
    screening_budget_fraction,
    verification_budget_fraction,
    use_conformal_bound,
    task_balanced,
):
    stages = {example.record_id: "base" for example in examples}
    screened = _learned_portfolio_kg_acquisitions(
        examples,
        stages,
        stage_estimates,
        evidence_estimates,
        model,
        calibrators,
        current_stage="base",
        next_stage="screening",
        execution_budget_fraction=execution_budget_fraction,
        acquisition_budget_fraction=screening_budget_fraction,
        candidate_indices=range(len(examples)),
        use_conformal_bound=use_conformal_bound,
        task_balanced=task_balanced,
    )
    for index in screened:
        stages[examples[index].record_id] = "screening"
    verified = _learned_portfolio_kg_acquisitions(
        examples,
        stages,
        stage_estimates,
        evidence_estimates,
        model,
        calibrators,
        current_stage="screening",
        next_stage="verification",
        execution_budget_fraction=execution_budget_fraction,
        acquisition_budget_fraction=verification_budget_fraction,
        candidate_indices=screened,
        use_conformal_bound=use_conformal_bound,
        task_balanced=task_balanced,
    )
    for index in verified:
        stages[examples[index].record_id] = "verification"
    return stages


def _learned_portfolio_kg_acquisitions(
    examples,
    stages,
    stage_estimates,
    evidence_estimates,
    model,
    calibrators,
    *,
    current_stage,
    next_stage,
    execution_budget_fraction,
    acquisition_budget_fraction,
    candidate_indices,
    use_conformal_bound,
    task_balanced,
):
    candidates = list(candidate_indices)
    if not candidates:
        return set()
    shadows = _leave_one_out_execution_shadows(
        examples,
        stages,
        stage_estimates,
        execution_budget_fraction,
        task_balanced=task_balanced,
    )
    transition = f"{current_stage}->{next_stage}"
    scores = []
    for index in candidates:
        example = examples[index]
        estimate = evidence_estimates[example.record_id][next_stage]
        structural = _decision_consistent_crossed_estimate(
            estimate,
            shadow_price=shadows[index],
        )
        current_score = _decision_score(
            example,
            current_stage,
            stage_estimates[example.record_id][current_stage],
        )
        features = portfolio_kg_features(
            current_score=current_score,
            shadow_price=shadows[index],
            structural_mean=structural.mean,
            structural_variance=structural.predictive_variance,
            internal_variance=structural.internal_variance,
            external_variance=structural.external_variance,
            interaction_variance=structural.interaction_variance,
            positive_probability=structural.positive_probability,
            evidence_cost=estimate.evidence_cost,
            future_score_cells=estimate.future_decision_score_cells,
        )
        prediction = model.predict(
            record_id=example.record_id,
            benchmark=example.benchmark,
            transition=transition,
            features=features,
        )
        score = prediction.expected_incremental_gain
        if use_conformal_bound:
            bound = calibrators[f"{example.benchmark}::{transition}"].predict_one(
                record_id=example.record_id,
                predicted_mean_gain=prediction.expected_incremental_gain,
                predicted_scale=math.sqrt(
                    max(1e-10, prediction.predictive_variance)
                ),
            )
            score = bound.lower_gain_bound
        scores.append((score, index))
    return _positive_budgeted_acquisitions(
        examples,
        scores,
        budget_fraction=acquisition_budget_fraction,
        task_balanced=task_balanced,
    )


def _decision_consistent_crossed_estimate(estimate, *, shadow_price):
    cells = counterfactual_evidence_value_cells(
        estimate.current_decision_score_by_model,
        estimate.future_decision_score_cells,
        evidence_cost=estimate.evidence_cost,
        cost_weight=COST_WEIGHT,
        current_abstention_value=shadow_price,
        future_abstention_value=shadow_price,
    )
    return crossed_numeric_estimate(cells)


def _decision_consistent_evidence_score(
    estimate,
    *,
    shadow_price,
    uncertainty_radius,
):
    crossed = _decision_consistent_crossed_estimate(
        estimate,
        shadow_price=shadow_price,
    )
    return crossed.mean - uncertainty_radius * math.sqrt(
        max(0.0, crossed.predictive_variance)
    )


def _leave_one_out_execution_shadows(
    examples,
    stages,
    stage_estimates,
    execution_budget_fraction,
    task_balanced=False,
):
    scores = [
        _decision_score(
            example,
            stages[example.record_id],
            stage_estimates[example.record_id][stages[example.record_id]],
        )
        for example in examples
    ]
    if not task_balanced:
        budget = max(1, math.ceil(execution_budget_fraction * len(scores)))
        return leave_one_out_topk_shadow_prices(scores, budget=budget, floor=0.0)
    shadows = [0.0] * len(examples)
    for benchmark in sorted({example.benchmark for example in examples}):
        indices = [
            index
            for index, example in enumerate(examples)
            if example.benchmark == benchmark
        ]
        local_scores = [scores[index] for index in indices]
        budget = max(1, math.ceil(execution_budget_fraction * len(indices)))
        local_shadows = leave_one_out_topk_shadow_prices(
            local_scores,
            budget=budget,
            floor=0.0,
        )
        for local_index, global_index in enumerate(indices):
            shadows[global_index] = local_shadows[local_index]
    return shadows


def _decision_score(example, stage, estimate):
    return estimate.gain_lower_bound + COST_WEIGHT * example.cumulative_evidence_cost(
        stage
    )


def _shadow_evidence_estimate(estimate, *, shadow_price, uncertainty_radius):
    cells = counterfactual_evidence_value_cells(
        estimate.current_gain_by_model,
        estimate.future_gain_cells,
        evidence_cost=estimate.evidence_cost,
        cost_weight=COST_WEIGHT,
        current_abstention_value=shadow_price,
        future_abstention_value=shadow_price,
    )
    crossed = crossed_numeric_estimate(cells)
    return crossed.mean - uncertainty_radius * math.sqrt(
        max(0.0, crossed.predictive_variance)
    )


def _execution_shadow_price(
    examples,
    stages,
    stage_estimates,
    execution_budget_fraction,
):
    scores = sorted(
        (
            stage_estimates[example.record_id][
                stages[example.record_id]
            ].gain_lower_bound
            for example in examples
        ),
        reverse=True,
    )
    budget = max(1, math.ceil(execution_budget_fraction * len(scores)))
    return max(0.0, scores[min(len(scores) - 1, budget - 1)])


def _evaluate_trajectory_policy(examples, stages, scores, *, budget_fraction):
    budget = max(1, math.ceil(budget_fraction * len(examples)))
    eligible = [index for index, score in enumerate(scores) if score > 0.0]
    eligible.sort(key=lambda index: (-scores[index], examples[index].record_id))
    selected = set(eligible[:budget])
    return _evaluate_selected_indices(examples, stages, selected)


def _evaluate_task_balanced_trajectory_policy(
    examples,
    stages,
    scores,
    *,
    budget_fraction,
):
    selected = set()
    for benchmark in sorted({example.benchmark for example in examples}):
        indices = [
            index
            for index, example in enumerate(examples)
            if example.benchmark == benchmark and scores[index] > 0.0
        ]
        budget = max(
            1,
            math.ceil(
                budget_fraction
                * sum(example.benchmark == benchmark for example in examples)
            ),
        )
        indices.sort(key=lambda index: (-scores[index], examples[index].record_id))
        selected.update(indices[:budget])
    return _evaluate_selected_indices(examples, stages, selected)


def _positive_budgeted_acquisitions(
    examples,
    scored_indices,
    *,
    budget_fraction,
    task_balanced,
):
    if not task_balanced:
        budget = max(1, math.ceil(budget_fraction * len(examples)))
        return {
            index
            for score, index in sorted(
                scored_indices,
                key=lambda row: (-row[0], examples[row[1]].record_id),
            )[:budget]
            if score > 0.0
        }
    selected = set()
    for benchmark in sorted({example.benchmark for example in examples}):
        rows = [
            (score, index)
            for score, index in scored_indices
            if examples[index].benchmark == benchmark
        ]
        task_count = sum(example.benchmark == benchmark for example in examples)
        budget = max(1, math.ceil(budget_fraction * task_count))
        selected.update(
            index
            for score, index in sorted(
                rows,
                key=lambda row: (-row[0], examples[row[1]].record_id),
            )[:budget]
            if score > 0.0
        )
    return selected


def _evaluate_selected_indices(examples, stages, selected):
    trajectory_values = []
    selected_gains = []
    evidence_cost = 0.0
    stage_counts = {stage: 0 for stage in STAGES}
    task = {}
    for index, example in enumerate(examples):
        stage = stages[example.record_id]
        stage_counts[stage] += 1
        cost = example.cumulative_evidence_cost(stage)
        evidence_cost += cost
        if index in selected:
            value = example.realized_gain(stage)
            selected_gains.append(value)
        else:
            value = -COST_WEIGHT * cost
        trajectory_values.append(value)
        row = task.setdefault(
            example.benchmark,
            {"count": 0, "selected": 0, "errors": 0, "total_gain": 0.0, "evidence_cost": 0.0},
        )
        row["count"] += 1
        row["evidence_cost"] += cost
        row["total_gain"] += value
        if index in selected:
            row["selected"] += 1
            row["errors"] += int(value <= 0.0)
    errors = sum(value <= 0.0 for value in selected_gains)
    for row in task.values():
        row["coverage"] = row["selected"] / row["count"]
        row["selective_risk"] = row["errors"] / row["selected"] if row["selected"] else 0.0
        row["population_mean_gain"] = row["total_gain"] / row["count"]
    task_rows = list(task.values())
    return {
        "selected_record_ids": sorted(examples[index].record_id for index in selected),
        "stage_by_record_id": {
            example.record_id: stages[example.record_id] for example in examples
        },
        "selected_count": len(selected),
        "coverage": len(selected) / len(examples),
        "errors": errors,
        "selective_risk": errors / len(selected) if selected else 0.0,
        "selected_mean_strict_gain": sum(selected_gains) / len(selected_gains) if selected_gains else 0.0,
        "selected_total_strict_gain": sum(selected_gains),
        "trajectory_total_strict_gain": sum(trajectory_values),
        "population_mean_strict_gain": sum(trajectory_values) / len(examples),
        "macro_task_selective_risk": sum(
            row["selective_risk"] for row in task_rows
        )
        / len(task_rows),
        "macro_task_population_mean_gain": sum(
            row["population_mean_gain"] for row in task_rows
        )
        / len(task_rows),
        "worst_task_selective_risk": max(
            row["selective_risk"] for row in task_rows
        ),
        "worst_task_population_mean_gain": min(
            row["population_mean_gain"] for row in task_rows
        ),
        "total_acquired_evidence_cost": evidence_cost,
        "stage_counts": stage_counts,
        "task_breakdown": task,
    }


def _oracle_evaluation(examples, *, budget_fraction):
    stages = {}
    scores = []
    for example in examples:
        best_stage = max(STAGES, key=lambda stage: example.realized_gain(stage))
        stages[example.record_id] = best_stage
        scores.append(max(0.0, example.realized_gain(best_stage)))
    return _evaluate_trajectory_policy(
        examples, stages, scores, budget_fraction=budget_fraction
    )


def _calibration_metrics(examples, estimates):
    output = {}
    for stage in STAGES:
        probabilities = [
            estimates[example.record_id][stage].calibrated_worthiness
            for example in examples
        ]
        labels = [int(example.realized_gain(stage) > 0.0) for example in examples]
        output[stage] = {
            "brier": sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels),
            "ece_10": _ece(probabilities, labels, bins=10),
            "positive_rate": sum(labels) / len(labels),
        }
    return output


def _static_reliability(example):
    features = example.base_record.get("features", {})
    return (
        0.25 * float(features.get("verbal_confidence", 0.5))
        + 0.25 * float(features.get("perturbation_stability", 0.5))
        + 0.20 * float(features.get("evidence_support", 0.5))
        + 0.15 * float(features.get("source_reliability", 0.5))
        + 0.15 * (1.0 - float(features.get("ood_score", 0.5)))
    )


def _feedback_partitions(examples, *, seed):
    partition_names = ("gain", "evidence", "policy")
    output = {}
    for benchmark in sorted({example.benchmark for example in examples}):
        task_rows = [example for example in examples if example.benchmark == benchmark]
        by_group = {}
        for example in task_rows:
            by_group.setdefault(example.group_id, []).append(example)
        groups = sorted(
            by_group,
            key=lambda group: (
                -len(by_group[group]),
                hashlib.sha256(
                    f"{seed}|h5-feedback|{benchmark}|{group}".encode()
                ).hexdigest(),
            ),
        )
        counts = {name: 0 for name in partition_names}
        for group in groups:
            partition = min(partition_names, key=lambda name: (counts[name], name))
            for example in by_group[group]:
                output[example.record_id] = partition
            counts[partition] += len(by_group[group])
    if len(output) != len(examples):
        raise RuntimeError("feedback partitioning missed records")
    return output


def _score_quantiles(values, probabilities):
    ordered = sorted(float(value) for value in values)
    output = []
    for probability in probabilities:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            output.append(ordered[lower])
        else:
            fraction = position - lower
            output.append(
                ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
            )
    return output


def _ece(probabilities, labels, *, bins):
    total = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not indices:
            continue
        confidence = sum(probabilities[index] for index in indices) / len(indices)
        accuracy = sum(labels[index] for index in indices) / len(indices)
        total += len(indices) / len(labels) * abs(confidence - accuracy)
    return total


def _clip_probability(value):
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
