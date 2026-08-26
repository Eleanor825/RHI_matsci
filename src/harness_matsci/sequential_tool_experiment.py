from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .continuous_gain_calibration import (
    ContinuousGainObservation,
    fit_continuous_gain_calibrator,
)
from .historical import (
    MAIN_MATERIAL_TASKS,
    grouped_four_way_fold,
    grouped_four_way_split,
    load_historical_tasks,
)
from .metrics import binary_metrics
from .risk_control import (
    audit_risk_controlled_grid,
    select_bounded_risk_grid,
    select_hoeffding_risk_grid,
    select_risk_controlled_grid,
)
from .scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    GainTarget,
    ScientificShortfallScales,
    gain_targets,
)
from .schema import ActionRecord
from .sequential_tool_benchmark import (
    RealScientificToolEnvironment,
    SequentialToolEnvironment,
    TrainOnlyStructuredDirectToolEnvironment,
    TrainOnlyStructuredSurrogateToolEnvironment,
    TrainOnlySurrogateToolEnvironment,
)
from .source_calibration import (
    ReplicaWorthinessObservation,
    SourceDecomposedWorthinessCalibrator,
    fit_source_decomposed_calibrator,
)
from .source_decomposition import decompose_source_variance


@dataclass(frozen=True)
class ReplicaPrediction:
    record_id: str
    benchmark: str
    mean_gain: float
    internal_variance: float
    external_variance: float
    tool_probability: float
    tool_margin: float
    tool_external_variance: float


class TaskLogisticCalibrator:
    def __init__(self, models_by_task: dict[str, Any]) -> None:
        self.models_by_task = models_by_task

    @classmethod
    def fit(cls, observations: list[ReplicaWorthinessObservation]) -> "TaskLogisticCalibrator":
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        models: dict[str, Any] = {}
        for task in sorted({observation.benchmark for observation in observations}):
            rows = [observation for observation in observations if observation.benchmark == task]
            labels = np.asarray([row.worthy for row in rows], dtype=int)
            if len(set(labels.tolist())) < 2:
                raise ValueError(f"task {task} requires both labels for calibration")
            matrix = np.asarray([[row.mean_gain] for row in rows], dtype=float)
            models[task] = LogisticRegression(C=10.0, max_iter=1000).fit(matrix, labels)
        return cls(models)

    def predict(self, observations: list[ReplicaWorthinessObservation]) -> list[float]:
        import numpy as np

        probabilities = []
        for observation in observations:
            model = self.models_by_task[observation.benchmark]
            probability = model.predict_proba(np.asarray([[observation.mean_gain]], dtype=float))[0, 1]
            probabilities.append(float(probability))
        return probabilities


class TaskToolProbabilityCalibrator:
    def __init__(self, models_by_task: dict[str, Any]) -> None:
        self.models_by_task = models_by_task

    @classmethod
    def fit(
        cls,
        predictions: list[ReplicaPrediction],
        labels_by_id: dict[str, int],
    ) -> "TaskToolProbabilityCalibrator":
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        models: dict[str, Any] = {}
        for task in sorted({prediction.benchmark for prediction in predictions}):
            rows = [prediction for prediction in predictions if prediction.benchmark == task]
            labels = np.asarray([labels_by_id[row.record_id] for row in rows], dtype=int)
            values = np.asarray([[_logit(row.tool_probability)] for row in rows], dtype=float)
            models[task] = LogisticRegression(C=10.0, max_iter=1000).fit(values, labels)
        return cls(models)

    def predict(self, predictions: list[ReplicaPrediction]) -> list[float]:
        import numpy as np

        output = []
        for prediction in predictions:
            model = self.models_by_task[prediction.benchmark]
            probability = model.predict_proba(
                np.asarray([[_logit(prediction.tool_probability)]], dtype=float)
            )[0, 1]
            output.append(float(probability))
        return output


class GainReplicaEnsemble:
    def __init__(
        self,
        feature_names_by_task: dict[str, list[str]],
        models_by_task: dict[str, list[Any]],
    ) -> None:
        self.feature_names_by_task = feature_names_by_task
        self.models_by_task = models_by_task

    @property
    def replica_count(self) -> int:
        counts = {len(models) for models in self.models_by_task.values()}
        if len(counts) != 1:
            raise ValueError("task-specific ensembles must have equal replica counts")
        return next(iter(counts))

    @classmethod
    def fit(
        cls,
        records: list[ActionRecord],
        targets: list[float],
        *,
        seed: int,
        replicas: int = 3,
        trees: int = 160,
    ) -> "GainReplicaEnsemble":
        import numpy as np
        from sklearn.ensemble import ExtraTreesRegressor

        if len(records) != len(targets):
            raise ValueError("training records and gain targets must align")
        feature_names_by_task: dict[str, list[str]] = {}
        models_by_task: dict[str, list[Any]] = {}
        tasks = sorted({record.benchmark for record in records})
        for task_index, task in enumerate(tasks):
            indices = [index for index, record in enumerate(records) if record.benchmark == task]
            task_records = [records[index] for index in indices]
            task_targets = np.asarray([targets[index] for index in indices], dtype=float)
            feature_names = _feature_names(task_records)
            matrix = _matrix(task_records, feature_names)
            models = []
            for replica in range(replicas):
                model = ExtraTreesRegressor(
                    n_estimators=trees,
                    min_samples_leaf=4,
                    max_features=0.8,
                    bootstrap=True,
                    max_samples=0.85,
                    random_state=seed * 1009 + task_index * 503 + replica * 97,
                    n_jobs=-1,
                )
                model.fit(matrix, task_targets)
                models.append(model)
            feature_names_by_task[task] = feature_names
            models_by_task[task] = models
        return cls(feature_names_by_task, models_by_task)

    def predict_matrix(self, evidence_records: list[ActionRecord]) -> list[list[float]]:
        output = [[0.0 for _ in evidence_records] for _ in range(self.replica_count)]
        for task, models in self.models_by_task.items():
            indices = [index for index, record in enumerate(evidence_records) if record.benchmark == task]
            if not indices:
                continue
            task_records = [evidence_records[index] for index in indices]
            matrix = _matrix(task_records, self.feature_names_by_task[task])
            for replica, model in enumerate(models):
                for index, value in zip(indices, model.predict(matrix)):
                    output[replica][index] = float(value)
        return output


def run_sequential_tool_experiment(
    raw_data_dir: str | Path,
    *,
    seed: int,
    budget_fraction: float = 0.10,
    evidence_replicas: tuple[int, ...] = (101, 211, 307),
    train_evidence_replicas: tuple[int, ...] = (11, 23, 47),
    alphas: tuple[float, ...] = (0.10, 0.20, 0.30),
    environment_kind: str = "semi_synthetic",
    regime_fold: int | None = None,
    n_regime_folds: int = 5,
    reservation_strategy: str = "outcome_conditioned_base_gain",
    train_fraction: float = 0.60,
    feedback_fraction: float = 0.15,
    acceptance_fraction: float = 0.10,
) -> dict[str, Any]:
    records_by_task = load_historical_tasks(raw_data_dir, MAIN_MATERIAL_TASKS)
    if regime_fold is None:
        splits_by_task = {
            task: grouped_four_way_split(
                records,
                seed=seed,
                train_fraction=train_fraction,
                feedback_fraction=feedback_fraction,
                acceptance_fraction=acceptance_fraction,
            )
            for task, records in records_by_task.items()
        }
        split_kind = "seeded_regime_subset"
    else:
        splits_by_task = {
            task: grouped_four_way_fold(
                records,
                fold_index=regime_fold,
                n_folds=n_regime_folds,
                train_fraction=train_fraction,
                feedback_fraction=feedback_fraction,
                acceptance_fraction=acceptance_fraction,
            )
            for task, records in records_by_task.items()
        }
        split_kind = "deterministic_group_covering_fold"
    splits = {
        split: [
            record
            for task in MAIN_MATERIAL_TASKS
            for record in splits_by_task[task][split]
        ]
        for split in ("train", "feedback", "acceptance", "test")
    }
    normalizer = EmpiricalUtilityNormalizer.fit(splits["train"])
    gain_config = GainConfig()
    reservations = BudgetReservationValues.fit(
        splits["train"],
        normalizer,
        gain_config,
        budget_fraction=budget_fraction,
        outside_option=0.0,
        strategy=reservation_strategy,
    )
    targets_by_split = {
        split: gain_targets(records, normalizer, gain_config, reservations)
        for split, records in splits.items()
    }
    target_by_id = {
        target.record_id: target
        for targets in targets_by_split.values()
        for target in targets
    }
    if environment_kind == "semi_synthetic":
        environment = SequentialToolEnvironment.fit(raw_data_dir, splits["train"])
    elif environment_kind == "train_only_surrogate":
        environment = TrainOnlySurrogateToolEnvironment.fit(
            raw_data_dir,
            splits["train"],
            environment_seed=seed,
        )
    elif environment_kind == "train_only_structured_surrogate":
        environment = TrainOnlyStructuredSurrogateToolEnvironment.fit(
            raw_data_dir,
            splits["train"],
            environment_seed=seed,
        )
    elif environment_kind == "train_only_structured_direct":
        environment = TrainOnlyStructuredDirectToolEnvironment.fit(
            raw_data_dir,
            splits["train"],
            environment_seed=seed,
        )
    elif environment_kind == "real_scientific_tools":
        environment = RealScientificToolEnvironment.fit(
            raw_data_dir,
            splits["train"],
            environment_seed=seed,
        )
    else:
        raise ValueError(f"unknown environment kind: {environment_kind}")
    nontraining_outcome_invariance_audit = (
        environment.audit_nontraining_outcome_invariance(
            splits["test"],
            evidence_replica=evidence_replicas[0],
            run_seed=seed + 1000,
        )
        if isinstance(environment, TrainOnlySurrogateToolEnvironment)
        else None
    )
    consistency = environment.label_consistency(
        record for records in splits.values() for record in records
    )
    stage_predictions: dict[str, dict[str, list[ReplicaPrediction]]] = {}
    stage_models: dict[str, GainReplicaEnsemble] = {}
    for stage in ("base", "screening", "verification"):
        train_observations = [
            environment.observe(
                record,
                stage=stage,
                evidence_replica=evidence_replica,
                run_seed=seed,
            ).record
            for record in splits["train"]
            for evidence_replica in train_evidence_replicas
        ]
        train_gains = [
            target_by_id[record.record_id].gain
            for record in splits["train"]
            for _ in train_evidence_replicas
        ]
        ensemble = GainReplicaEnsemble.fit(train_observations, train_gains, seed=seed + len(stage))
        stage_models[stage] = ensemble
        stage_predictions[stage] = {
            split: _predict_replicas(
                ensemble,
                environment,
                records,
                stage=stage,
                evidence_replicas=evidence_replicas,
                run_seed=seed + 1000,
            )
            for split, records in splits.items()
            if split != "train"
        }
    worthiness_by_id = {
        target.record_id: target.worthy
        for targets in targets_by_split.values()
        for target in targets
    }
    methods: dict[str, dict[str, list[float]]] = {}
    method_metadata: dict[str, dict[str, Any]] = {}
    baseline_raw_scores = {
        "base_verbal_confidence": {
            split: [float(record.features.get("verbal_confidence", 0.5)) for record in splits[split]]
            for split in ("feedback", "acceptance", "test")
        },
        "base_evidence_heuristic": {
            split: [
                max(
                    0.0,
                    min(
                        1.0,
                        float(record.features.get("evidence_support", 0.0))
                        - float(record.features.get("evidence_conflict", 0.0)),
                    ),
                )
                for record in splits[split]
            ]
            for split in ("feedback", "acceptance", "test")
        },
        "base_ensemble_lcb": {
            split: [
                prediction.mean_gain
                - math.sqrt(
                    max(0.0, prediction.internal_variance + prediction.external_variance)
                )
                for prediction in stage_predictions["base"][split]
            ]
            for split in ("feedback", "acceptance", "test")
        },
    }
    for method, raw_scores_by_split in baseline_raw_scores.items():
        calibrator = TaskLogisticCalibrator.fit(
            _scalar_calibration_observations(
                splits["feedback"],
                raw_scores_by_split["feedback"],
                worthiness_by_id,
            )
        )
        methods[method] = {
            split: calibrator.predict(
                _scalar_calibration_observations(
                    splits[split], raw_scores, worthiness_by_id
                )
            )
            for split, raw_scores in raw_scores_by_split.items()
        }
        method_metadata[method] = {
            "acquisition_policy": "none",
            "acquisition_counts": {
                split: 0 for split in ("feedback", "acceptance", "test")
            },
            "baseline_family": method.removeprefix("base_"),
        }
    for stage in ("base", "screening", "verification"):
        feedback_observations = _calibration_observations(
            stage_predictions[stage]["feedback"], worthiness_by_id
        )
        mean_calibrator = TaskLogisticCalibrator.fit(feedback_observations)
        methods[f"{stage}_mean"] = {
            split: mean_calibrator.predict(
                _calibration_observations(predictions, worthiness_by_id)
            )
            for split, predictions in stage_predictions[stage].items()
        }
        method_metadata[f"{stage}_mean"] = {
            "acquisition_policy": "all_actions" if stage != "base" else "none",
            "acquisition_counts": {
                split: len(predictions) if stage != "base" else 0
                for split, predictions in stage_predictions[stage].items()
            },
        }
    verification_feedback = _calibration_observations(
        stage_predictions["verification"]["feedback"], worthiness_by_id
    )
    source_calibrator = fit_source_decomposed_calibrator(verification_feedback)
    methods["sd_worth"] = {
        split: source_calibrator.predict(
            _calibration_observations(predictions, worthiness_by_id)
        )
        for split, predictions in stage_predictions["verification"].items()
    }
    method_metadata["sd_worth"] = {
        "acquisition_policy": "all_actions",
        "acquisition_counts": {
            split: len(predictions)
            for split, predictions in stage_predictions["verification"].items()
        },
    }
    tool_calibrator = TaskToolProbabilityCalibrator.fit(
        stage_predictions["verification"]["feedback"], worthiness_by_id
    )
    methods["tool_probability"] = {
        split: tool_calibrator.predict(predictions)
        for split, predictions in stage_predictions["verification"].items()
    }
    method_metadata["tool_probability"] = {
        "acquisition_policy": "all_actions",
        "acquisition_counts": {
            split: len(predictions)
            for split, predictions in stage_predictions["verification"].items()
        },
    }
    continuous_feedback = _continuous_observations(
        stage_predictions["verification"]["feedback"], target_by_id
    )
    continuous_source_calibrator = fit_continuous_gain_calibrator(
        continuous_feedback,
        use_source_variance=True,
    )
    continuous_mean_calibrator = fit_continuous_gain_calibrator(
        continuous_feedback,
        use_source_variance=False,
    )
    for method, calibrator in (
        ("continuous_mean", continuous_mean_calibrator),
        ("continuous_sd_worth", continuous_source_calibrator),
    ):
        methods[method] = {
            split: [
                prediction.action_worthiness
                for prediction in calibrator.predict(
                    _continuous_observations(predictions, target_by_id)
                )
            ]
            for split, predictions in stage_predictions["verification"].items()
        }
        method_metadata[method] = {
            "acquisition_policy": "all_actions",
            "acquisition_counts": {
                split: len(predictions)
                for split, predictions in stage_predictions["verification"].items()
            },
            "continuous_gain_calibrator": calibrator.to_json(),
        }
    direct_feedback = _tool_margin_continuous_observations(
        stage_predictions["verification"]["feedback"],
        stage_predictions["base"]["feedback"],
        target_by_id,
    )
    direct_source_calibrator = fit_continuous_gain_calibrator(
        direct_feedback,
        use_source_variance=True,
    )
    direct_mean_calibrator = fit_continuous_gain_calibrator(
        direct_feedback,
        use_source_variance=False,
    )
    for method, calibrator in (
        ("tool_margin_continuous_mean", direct_mean_calibrator),
        ("tool_margin_continuous_sd_worth", direct_source_calibrator),
    ):
        methods[method] = {
            split: [
                prediction.action_worthiness
                for prediction in calibrator.predict(
                    _tool_margin_continuous_observations(
                        stage_predictions["verification"][split],
                        stage_predictions["base"][split],
                        target_by_id,
                    )
                )
            ]
            for split in stage_predictions["verification"]
        }
        method_metadata[method] = {
            "acquisition_policy": "all_actions",
            "acquisition_counts": {
                split: len(predictions)
                for split, predictions in stage_predictions["verification"].items()
            },
            "continuous_gain_calibrator": calibrator.to_json(),
            "direct_tool_margin": True,
        }
    labels = {
        split: [target.worthy for target in targets_by_split[split]]
        for split in ("feedback", "acceptance", "test")
    }
    gains = {
        split: [target.gain for target in targets_by_split[split]]
        for split in ("feedback", "acceptance", "test")
    }
    shortfall_scales = ScientificShortfallScales.fit(targets_by_split["train"])
    shortfall_losses = {
        split: [shortfall_scales.loss(target) for target in targets_by_split[split]]
        for split in ("feedback", "acceptance", "test")
    }
    for source_method in (
        "verification_mean",
        "sd_worth",
        "tool_probability",
        "continuous_mean",
        "continuous_sd_worth",
        "tool_margin_continuous_mean",
        "tool_margin_continuous_sd_worth",
    ):
        sparse_method = f"sparse_{source_method}"
        acquisition = _fit_sparse_acquisition(
            methods["base_mean"]["feedback"],
            methods[source_method]["feedback"],
            gains["feedback"],
            primary_tool_cost=0.002,
        )
        sparse_probabilities: dict[str, list[float]] = {}
        acquisition_counts: dict[str, int] = {}
        for split in ("feedback", "acceptance", "test"):
            sparse_scores, acquired = _apply_sparse_acquisition(
                methods["base_mean"][split],
                methods[source_method][split],
                threshold=acquisition["base_score_threshold"],
            )
            sparse_probabilities[split] = sparse_scores
            acquisition_counts[split] = acquired
        methods[sparse_method] = sparse_probabilities
        method_metadata[sparse_method] = {
            "acquisition_policy": "feedback_selected_base_score_threshold",
            "source_method": source_method,
            "primary_tool_cost": 0.002,
            "acquisition": acquisition,
            "acquisition_counts": acquisition_counts,
        }
    for source_method in (
        "continuous_mean",
        "continuous_sd_worth",
        "tool_margin_continuous_mean",
        "tool_margin_continuous_sd_worth",
    ):
        sparse_method = f"task_sparse_{source_method}"
        acquisition = _fit_task_sparse_acquisition(
            methods["base_mean"]["feedback"],
            methods[source_method]["feedback"],
            gains["feedback"],
            splits["feedback"],
            primary_tool_cost=0.002,
        )
        sparse_probabilities: dict[str, list[float]] = {}
        acquisition_counts: dict[str, int] = {}
        acquisition_counts_by_task: dict[str, dict[str, int]] = {}
        for split in ("feedback", "acceptance", "test"):
            sparse_scores, counts = _apply_task_sparse_acquisition(
                methods["base_mean"][split],
                methods[source_method][split],
                splits[split],
                thresholds_by_task={
                    task: values["base_score_threshold"]
                    for task, values in acquisition.items()
                },
            )
            sparse_probabilities[split] = sparse_scores
            acquisition_counts[split] = sum(counts.values())
            acquisition_counts_by_task[split] = counts
        methods[sparse_method] = sparse_probabilities
        method_metadata[sparse_method] = {
            "acquisition_policy": "feedback_selected_task_conditional_base_threshold",
            "source_method": source_method,
            "primary_tool_cost": 0.002,
            "acquisition_by_task": acquisition,
            "acquisition_counts": acquisition_counts,
            "acquisition_counts_by_task": acquisition_counts_by_task,
        }
    joint_certificate_aware = None
    safe_activation = None
    if environment_kind == "real_scientific_tools":
        joint_certificate_aware = {
            "source_decomposed": _run_joint_certificate_aware_policy(
                methods["base_mean"],
                methods["tool_margin_continuous_sd_worth"],
                labels,
                gains,
                shortfall_losses,
                splits,
                alpha=0.20,
                delta=0.05,
                tool_cost=0.002,
            ),
            "no_source": _run_joint_certificate_aware_policy(
                methods["base_mean"],
                methods["tool_margin_continuous_mean"],
                labels,
                gains,
                shortfall_losses,
                splits,
                alpha=0.20,
                delta=0.05,
                tool_cost=0.002,
            ),
        }
        safe_activation = {
            "source_decomposed": _run_safe_activation_policy(
                methods["task_sparse_tool_margin_continuous_sd_worth"],
                method_metadata["task_sparse_tool_margin_continuous_sd_worth"],
                labels,
                gains,
                shortfall_losses,
                splits,
                alpha=0.20,
                delta=0.05,
            ),
            "no_source": _run_safe_activation_policy(
                methods["task_sparse_tool_margin_continuous_mean"],
                method_metadata["task_sparse_tool_margin_continuous_mean"],
                labels,
                gains,
                shortfall_losses,
                splits,
                alpha=0.20,
                delta=0.05,
            ),
        }
    method_results = {}
    for method, probabilities_by_split in methods.items():
        stage = _method_stage(method)
        metadata = method_metadata[method]
        tool_cost = (
            float(metadata.get("primary_tool_cost", 0.0))
            if "sparse_" in method
            else environment.stage_specs[stage].cost
        )
        calibration_metrics = binary_metrics(labels["test"], probabilities_by_split["test"], 0.5)
        risk_results = {}
        risk_grid_diagnostics = {}
        for alpha in alphas:
            risk_grid_diagnostics[f"alpha_{alpha:.2f}"] = audit_risk_controlled_grid(
                probabilities_by_split["acceptance"],
                labels["acceptance"],
                coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=alpha,
                delta=0.05,
            )
            selection = select_risk_controlled_grid(
                probabilities_by_split["acceptance"],
                labels["acceptance"],
                coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=alpha,
                delta=0.05,
            )
            risk_results[f"alpha_{alpha:.2f}"] = {
                "acceptance": selection.to_json(),
                "test": _evaluate_threshold(
                    probabilities_by_split["test"],
                    labels["test"],
                    gains["test"],
                    splits["test"],
                    threshold=selection.threshold,
                    tool_cost=tool_cost,
                    acquired_count=int(metadata["acquisition_counts"]["test"]),
                    shortfall_losses=shortfall_losses["test"],
                ),
            }
        shortfall_risk_results = {}
        hoeffding_shortfall_risk_results = {}
        for alpha in alphas:
            selection = select_bounded_risk_grid(
                probabilities_by_split["acceptance"],
                shortfall_losses["acceptance"],
                coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=alpha,
                delta=0.05,
            )
            shortfall_risk_results[f"alpha_{alpha:.2f}"] = {
                "acceptance": selection.to_json(),
                "test": _evaluate_threshold(
                    probabilities_by_split["test"],
                    labels["test"],
                    gains["test"],
                    splits["test"],
                    threshold=selection.threshold,
                    tool_cost=tool_cost,
                    acquired_count=int(metadata["acquisition_counts"]["test"]),
                    shortfall_losses=shortfall_losses["test"],
                ),
            }
            hoeffding_selection = select_hoeffding_risk_grid(
                probabilities_by_split["acceptance"],
                shortfall_losses["acceptance"],
                coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=alpha,
                delta=0.05,
            )
            hoeffding_shortfall_risk_results[f"alpha_{alpha:.2f}"] = {
                "acceptance": hoeffding_selection.to_json(),
                "test": _evaluate_threshold(
                    probabilities_by_split["test"],
                    labels["test"],
                    gains["test"],
                    splits["test"],
                    threshold=hoeffding_selection.threshold,
                    tool_cost=tool_cost,
                    acquired_count=int(metadata["acquisition_counts"]["test"]),
                    shortfall_losses=shortfall_losses["test"],
                ),
            }
        task_risk_results = {}
        task_shortfall_risk_results = {}
        for alpha in alphas:
            binary_thresholds, binary_acceptance = _task_conditional_thresholds(
                probabilities_by_split["acceptance"],
                labels["acceptance"],
                splits["acceptance"],
                alpha=alpha,
                delta=0.05,
                bounded=False,
            )
            shortfall_thresholds, shortfall_acceptance = _task_conditional_thresholds(
                probabilities_by_split["acceptance"],
                shortfall_losses["acceptance"],
                splits["acceptance"],
                alpha=alpha,
                delta=0.05,
                bounded=True,
            )
            task_risk_results[f"alpha_{alpha:.2f}"] = {
                "acceptance": binary_acceptance,
                "test": _evaluate_threshold(
                    probabilities_by_split["test"],
                    labels["test"],
                    gains["test"],
                    splits["test"],
                    threshold=binary_thresholds,
                    tool_cost=tool_cost,
                    acquired_count=int(metadata["acquisition_counts"]["test"]),
                    shortfall_losses=shortfall_losses["test"],
                ),
            }
            task_shortfall_risk_results[f"alpha_{alpha:.2f}"] = {
                "acceptance": shortfall_acceptance,
                "test": _evaluate_threshold(
                    probabilities_by_split["test"],
                    labels["test"],
                    gains["test"],
                    splits["test"],
                    threshold=shortfall_thresholds,
                    tool_cost=tool_cost,
                    acquired_count=int(metadata["acquisition_counts"]["test"]),
                    shortfall_losses=shortfall_losses["test"],
                ),
            }
        method_results[method] = {
            "stage": stage,
            "tool_cost": tool_cost,
            "method_metadata": metadata,
            "calibration": calibration_metrics,
            "risk_control": risk_results,
            "risk_grid_diagnostics": risk_grid_diagnostics,
            "shortfall_risk_control": shortfall_risk_results,
            "hoeffding_shortfall_risk_control": hoeffding_shortfall_risk_results,
            "task_conditional_risk_control": task_risk_results,
            "task_conditional_shortfall_risk_control": task_shortfall_risk_results,
        }
    harness_evolution = _run_harness_evolution(
        methods,
        method_metadata,
        labels,
        gains,
        shortfall_losses,
        splits,
        alpha=0.20,
        delta=0.05,
    )
    harness_evolution_hoeffding_shortfall = _run_harness_evolution(
        methods,
        method_metadata,
        labels,
        gains,
        shortfall_losses,
        splits,
        alpha=0.20,
        delta=0.05,
        risk_family="hoeffding_shortfall",
    )
    return {
        "experiment": (
            "h9_1_structured_surrogate_tools"
            if environment_kind == "train_only_structured_surrogate"
            else "h10_real_scientific_tools"
            if environment_kind == "real_scientific_tools"
            else "h9_2_direct_tool_gain"
            if environment_kind == "train_only_structured_direct"
            else "h9_train_only_surrogate_tools"
            if environment_kind == "train_only_surrogate"
            else "h4_sequential_tool_benchmark_v2"
        ),
        "environment_kind": environment_kind,
        "nontraining_outcome_invariance_audit": nontraining_outcome_invariance_audit,
        "seed": seed,
        "regime_fold": regime_fold,
        "n_regime_folds": n_regime_folds if regime_fold is not None else None,
        "split_kind": split_kind,
        "source_split_fractions": {
            "train": train_fraction,
            "feedback": feedback_fraction,
            "acceptance": acceptance_fraction,
        },
        "tasks": list(MAIN_MATERIAL_TASKS),
        "test_groups_by_task": {
            task: sorted(
                {
                    str(record.metadata.get("group_id", record.benchmark))
                    for record in task_splits["test"]
                }
            )
            for task, task_splits in splits_by_task.items()
        },
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "task_split_sizes": {
            task: {split: len(records) for split, records in task_splits.items()}
            for task, task_splits in splits_by_task.items()
        },
        "budget_fraction": budget_fraction,
        "reservation_values": reservations.to_json(),
        "unique_training_threshold": environment.unique_discovery_threshold,
        "label_margin_consistency": consistency,
        "replica_contract": {
            "internal_replicas": 3,
            "external_replicas": len(evidence_replicas),
            "complete_matrix": True,
            "external_replica_ids": list(evidence_replicas),
        },
        "source_calibrator": source_calibrator.to_json(),
        "continuous_source_calibrator": continuous_source_calibrator.to_json(),
        "shortfall_scales": shortfall_scales.to_json(),
        "methods": method_results,
        "harness_evolution": harness_evolution,
        "harness_evolution_hoeffding_shortfall": harness_evolution_hoeffding_shortfall,
        "joint_certificate_aware_acquisition": joint_certificate_aware,
        "safe_activation": safe_activation,
    }


def _predict_replicas(
    ensemble: GainReplicaEnsemble,
    environment: SequentialToolEnvironment,
    records: list[ActionRecord],
    *,
    stage: str,
    evidence_replicas: tuple[int, ...],
    run_seed: int,
) -> list[ReplicaPrediction]:
    observations_by_replica = [
        [
            environment.observe(
                record,
                stage=stage,
                evidence_replica=evidence_replica,
                run_seed=run_seed,
            )
            for record in records
        ]
        for evidence_replica in evidence_replicas
    ]
    model_predictions_by_evidence = [
        ensemble.predict_matrix([observation.record for observation in observations])
        for observations in observations_by_replica
    ]
    output = []
    for record_index, record in enumerate(records):
        matrix = [
            [
                model_predictions_by_evidence[evidence_index][model_index][record_index]
                for evidence_index in range(len(evidence_replicas))
            ]
            for model_index in range(ensemble.replica_count)
        ]
        variance = decompose_source_variance(matrix)
        mean_gain = mean(value for row in matrix for value in row)
        tool_probability = mean(
            float(observations_by_replica[index][record_index].record.features.get("tool_positive_probability", 0.5))
            for index in range(len(evidence_replicas))
        )
        tool_margins = [
            float(
                observations_by_replica[index][record_index].record.features.get(
                    "tool_estimated_margin", 0.0
                )
            )
            for index in range(len(evidence_replicas))
        ]
        tool_margin = mean(tool_margins)
        tool_external_variance = mean(
            (value - tool_margin) ** 2 for value in tool_margins
        )
        output.append(
            ReplicaPrediction(
                record_id=record.record_id,
                benchmark=record.benchmark,
                mean_gain=mean_gain,
                internal_variance=variance.internal_variance,
                external_variance=variance.external_variance,
                tool_probability=tool_probability,
                tool_margin=tool_margin,
                tool_external_variance=tool_external_variance,
            )
        )
    return output


def _calibration_observations(
    predictions: list[ReplicaPrediction],
    labels_by_id: dict[str, int],
) -> list[ReplicaWorthinessObservation]:
    return [
        ReplicaWorthinessObservation(
            record_id=prediction.record_id,
            benchmark=prediction.benchmark,
            mean_gain=prediction.mean_gain,
            internal_variance=prediction.internal_variance,
            external_variance=prediction.external_variance,
            worthy=labels_by_id[prediction.record_id],
        )
        for prediction in predictions
    ]


def _scalar_calibration_observations(
    records: list[ActionRecord],
    scores: list[float],
    labels_by_id: dict[str, int],
) -> list[ReplicaWorthinessObservation]:
    if len(records) != len(scores):
        raise ValueError("records and scalar scores must align")
    return [
        ReplicaWorthinessObservation(
            record_id=record.record_id,
            benchmark=record.benchmark,
            mean_gain=float(score),
            internal_variance=0.0,
            external_variance=0.0,
            worthy=labels_by_id[record.record_id],
        )
        for record, score in zip(records, scores)
    ]


def _continuous_observations(
    predictions: list[ReplicaPrediction],
    targets_by_id: dict[str, GainTarget],
) -> list[ContinuousGainObservation]:
    return [
        ContinuousGainObservation(
            record_id=prediction.record_id,
            benchmark=prediction.benchmark,
            mean_gain=prediction.mean_gain,
            internal_variance=prediction.internal_variance,
            external_variance=prediction.external_variance,
            gain=targets_by_id[prediction.record_id].gain,
        )
        for prediction in predictions
    ]


def _tool_margin_continuous_observations(
    tool_predictions: list[ReplicaPrediction],
    base_predictions: list[ReplicaPrediction],
    targets_by_id: dict[str, GainTarget],
) -> list[ContinuousGainObservation]:
    if len(tool_predictions) != len(base_predictions):
        raise ValueError("tool and base predictions must align")
    observations = []
    for tool, base in zip(tool_predictions, base_predictions):
        if tool.record_id != base.record_id:
            raise ValueError("tool and base prediction IDs must align")
        observations.append(
            ContinuousGainObservation(
                record_id=tool.record_id,
                benchmark=tool.benchmark,
                mean_gain=tool.tool_margin,
                internal_variance=base.internal_variance,
                external_variance=tool.tool_external_variance,
                gain=targets_by_id[tool.record_id].gain,
            )
        )
    return observations


def _evaluate_threshold(
    probabilities: list[float],
    labels: list[int],
    gains: list[float],
    records: list[ActionRecord],
    *,
    threshold: float | dict[str, float],
    tool_cost: float,
    acquired_count: int | None = None,
    shortfall_losses: list[float] | None = None,
) -> dict[str, Any]:
    if isinstance(threshold, dict):
        selected = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= threshold[records[index].benchmark]
        ]
    else:
        selected = [index for index, probability in enumerate(probabilities) if probability >= threshold]
    selected_count = len(selected)
    selected_gain = sum(gains[index] for index in selected)
    if shortfall_losses is not None and len(shortfall_losses) != len(labels):
        raise ValueError("shortfall losses must align with labels")
    task_risks = {}
    for task in MAIN_MATERIAL_TASKS:
        task_selected = [index for index in selected if records[index].benchmark == task]
        if task_selected:
            task_risks[task] = sum(1 - labels[index] for index in task_selected) / len(task_selected)
    regime_risks = {}
    regimes = sorted({str(records[index].metadata.get("group_id", records[index].benchmark)) for index in selected})
    for regime in regimes:
        regime_selected = [
            index
            for index in selected
            if str(records[index].metadata.get("group_id", records[index].benchmark)) == regime
        ]
        if len(regime_selected) >= 5:
            regime_risks[regime] = sum(1 - labels[index] for index in regime_selected) / len(regime_selected)
    if acquired_count is None:
        acquired_count = len(labels) if tool_cost > 0.0 else 0
    total_acquisition_cost = tool_cost * acquired_count
    break_even_cost = selected_gain / acquired_count if acquired_count else None
    cost_sensitivity = {
        f"cost_{cost:.4f}": (selected_gain - cost * acquired_count) / len(labels)
        for cost in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.010)
    }
    return {
        "threshold": threshold,
        "selected_count": selected_count,
        "coverage": selected_count / len(labels),
        "selective_risk": (
            sum(1 - labels[index] for index in selected) / selected_count if selected_count else 0.0
        ),
        "selected_mean_gain": selected_gain / selected_count if selected_count else 0.0,
        "selected_mean_gain_after_tool_cost": (
            selected_gain / selected_count - tool_cost if selected_count else 0.0
        ),
        "selected_mean_shortfall": (
            sum(shortfall_losses[index] for index in selected) / selected_count
            if selected_count and shortfall_losses is not None
            else 0.0
        ),
        "acquired_count": acquired_count,
        "acquisition_fraction": acquired_count / len(labels),
        "total_acquisition_cost": total_acquisition_cost,
        "population_net_gain": (selected_gain - total_acquisition_cost) / len(labels),
        "population_net_gain_all_actions_acquire_tool": (
            selected_gain - tool_cost * len(labels)
        ) / len(labels),
        "break_even_tool_cost_per_acquired_action": break_even_cost,
        "cost_sensitivity": cost_sensitivity,
        "worst_task_risk": max(task_risks.values()) if task_risks else 0.0,
        "task_risks": task_risks,
        "worst_regime_risk_min5": max(regime_risks.values()) if regime_risks else 0.0,
        "regimes_with_min5_selected": len(regime_risks),
    }


def _feature_names(records: list[ActionRecord]) -> list[str]:
    numeric = {name for record in records for name in record.features}
    return sorted(numeric)


def _matrix(records: list[ActionRecord], feature_names: list[str]):
    import numpy as np

    return np.asarray(
        [
            [
                float(record.features.get(name, 0.0))
                for name in feature_names
            ]
            for record in records
        ],
        dtype=float,
    )


def _method_stage(method: str) -> str:
    if method.startswith("base"):
        return "base"
    if method.startswith("screening"):
        return "screening"
    return "verification"


def _fit_sparse_acquisition(
    base_scores: list[float],
    verification_scores: list[float],
    gains: list[float],
    *,
    primary_tool_cost: float,
) -> dict[str, float]:
    if not (len(base_scores) == len(verification_scores) == len(gains)):
        raise ValueError("sparse acquisition inputs must align")
    order = sorted(range(len(base_scores)), key=lambda index: (-base_scores[index], index))
    best: dict[str, float] | None = None
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        count = max(1, min(len(order), math.ceil(fraction * len(order))))
        acquired = order[:count]
        executed = [index for index in acquired if verification_scores[index] >= 0.5]
        net_gain = sum(gains[index] for index in executed) - primary_tool_cost * count
        next_score = base_scores[order[count]] if count < len(order) else -1.0
        threshold = (base_scores[order[count - 1]] + next_score) / 2.0
        candidate = {
            "requested_fraction": fraction,
            "feedback_acquired_count": float(count),
            "feedback_executed_count": float(len(executed)),
            "feedback_total_net_gain": net_gain,
            "base_score_threshold": threshold,
        }
        if best is None or (candidate["feedback_total_net_gain"], -fraction) > (
            best["feedback_total_net_gain"],
            -best["requested_fraction"],
        ):
            best = candidate
    if best is None:
        raise RuntimeError("failed to fit sparse acquisition policy")
    return best


def _apply_sparse_acquisition(
    base_scores: list[float],
    verification_scores: list[float],
    *,
    threshold: float,
) -> tuple[list[float], int]:
    output = []
    acquired = 0
    for base_score, verification_score in zip(base_scores, verification_scores):
        if base_score >= threshold:
            acquired += 1
            output.append(verification_score)
        else:
            output.append(0.0)
    return output, acquired


def _fit_task_sparse_acquisition(
    base_scores: list[float],
    verification_scores: list[float],
    gains: list[float],
    records: list[ActionRecord],
    *,
    primary_tool_cost: float,
) -> dict[str, dict[str, float]]:
    output = {}
    for task in MAIN_MATERIAL_TASKS:
        indices = [index for index, record in enumerate(records) if record.benchmark == task]
        output[task] = _fit_sparse_acquisition(
            [base_scores[index] for index in indices],
            [verification_scores[index] for index in indices],
            [gains[index] for index in indices],
            primary_tool_cost=primary_tool_cost,
        )
    return output


def _apply_task_sparse_acquisition(
    base_scores: list[float],
    verification_scores: list[float],
    records: list[ActionRecord],
    *,
    thresholds_by_task: dict[str, float],
) -> tuple[list[float], dict[str, int]]:
    output = []
    counts = {task: 0 for task in MAIN_MATERIAL_TASKS}
    for base_score, verification_score, record in zip(base_scores, verification_scores, records):
        if base_score >= thresholds_by_task[record.benchmark]:
            counts[record.benchmark] += 1
            output.append(verification_score)
        else:
            output.append(0.0)
    return output, counts


def _task_conditional_thresholds(
    scores: list[float],
    outcomes: list[int] | list[float],
    records: list[ActionRecord],
    *,
    alpha: float,
    delta: float,
    bounded: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = {}
    selections = {}
    for task in MAIN_MATERIAL_TASKS:
        indices = [index for index, record in enumerate(records) if record.benchmark == task]
        task_scores = [scores[index] for index in indices]
        task_outcomes = [outcomes[index] for index in indices]
        if bounded:
            selection = select_bounded_risk_grid(
                task_scores,
                [float(value) for value in task_outcomes],
                coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=alpha,
                delta=delta / len(MAIN_MATERIAL_TASKS),
            )
        else:
            selection = select_risk_controlled_grid(
                task_scores,
                [int(value) for value in task_outcomes],
                coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                alpha=alpha,
                delta=delta / len(MAIN_MATERIAL_TASKS),
            )
        thresholds[task] = selection.threshold
        selections[task] = selection.to_json()
    return thresholds, {
        "by_task": selections,
        "simultaneous_delta": delta,
        "per_task_delta": delta / len(MAIN_MATERIAL_TASKS),
        "bounded_loss": bounded,
    }


def _run_harness_evolution(
    methods: dict[str, dict[str, list[float]]],
    method_metadata: dict[str, dict[str, Any]],
    labels: dict[str, list[int]],
    gains: dict[str, list[float]],
    shortfall_losses: dict[str, list[float]],
    splits: dict[str, list[ActionRecord]],
    *,
    alpha: float,
    delta: float,
    risk_family: str = "binary_failure",
) -> dict[str, Any]:
    if risk_family not in {"binary_failure", "hoeffding_shortfall"}:
        raise ValueError(f"unsupported evolution risk family: {risk_family}")
    candidates = (
        "base_mean",
        "sparse_verification_mean",
        "sparse_tool_probability",
        "task_sparse_continuous_mean",
        "task_sparse_continuous_sd_worth",
    )
    per_harness_delta = delta / len(candidates)
    evaluated = {}
    for method in candidates:
        probabilities = methods[method]
        metadata = method_metadata[method]
        tool_cost = (
            float(metadata.get("primary_tool_cost", 0.0))
            if "sparse_" in method
            else 0.0
        )
        selection_function = (
            select_risk_controlled_grid
            if risk_family == "binary_failure"
            else select_hoeffding_risk_grid
        )
        acceptance_losses = (
            labels["acceptance"]
            if risk_family == "binary_failure"
            else shortfall_losses["acceptance"]
        )
        selection = selection_function(
            probabilities["acceptance"],
            acceptance_losses,
            coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
            alpha=alpha,
            delta=per_harness_delta,
        )
        acceptance = _evaluate_threshold(
            probabilities["acceptance"],
            labels["acceptance"],
            gains["acceptance"],
            splits["acceptance"],
            threshold=selection.threshold,
            tool_cost=tool_cost,
            acquired_count=int(metadata["acquisition_counts"]["acceptance"]),
            shortfall_losses=shortfall_losses["acceptance"],
        )
        test = _evaluate_threshold(
            probabilities["test"],
            labels["test"],
            gains["test"],
            splits["test"],
            threshold=selection.threshold,
            tool_cost=tool_cost,
            acquired_count=int(metadata["acquisition_counts"]["test"]),
            shortfall_losses=shortfall_losses["test"],
        )
        evaluated[method] = {
            "risk_selection": selection.to_json(),
            "acceptance": acceptance,
            "test": test,
        }
    incumbent = "base_mean"
    incumbent_gain = evaluated[incumbent]["acceptance"]["population_net_gain"]
    history = [
        {
            "round": 0,
            "candidate": incumbent,
            "accepted": True,
            "acceptance_population_net_gain": incumbent_gain,
            "reason": "initial_harness",
        }
    ]
    for round_index, candidate in enumerate(candidates[1:], 1):
        candidate_result = evaluated[candidate]
        candidate_gain = candidate_result["acceptance"]["population_net_gain"]
        has_coverage = candidate_result["risk_selection"]["selected_count"] > 0
        accepted = has_coverage and candidate_gain > incumbent_gain
        history.append(
            {
                "round": round_index,
                "candidate": candidate,
                "incumbent_before": incumbent,
                "accepted": accepted,
                "certified_nonzero_coverage": has_coverage,
                "acceptance_population_net_gain": candidate_gain,
                "incumbent_population_net_gain": incumbent_gain,
                "reason": (
                    "higher_certified_acceptance_net_gain"
                    if accepted
                    else "no_certified_net_gain_improvement"
                ),
            }
        )
        if accepted:
            incumbent = candidate
            incumbent_gain = candidate_gain
    return {
        "risk_family": risk_family,
        "alpha": alpha,
        "family_wise_delta": delta,
        "per_harness_delta": per_harness_delta,
        "candidate_order": list(candidates),
        "history": history,
        "selected_harness": incumbent,
        "selected_acceptance_population_net_gain": incumbent_gain,
        "selected_result": evaluated[incumbent],
        "all_candidates": evaluated,
    }


def _run_joint_certificate_aware_policy(
    base_scores: dict[str, list[float]],
    verification_scores: dict[str, list[float]],
    labels: dict[str, list[int]],
    gains: dict[str, list[float]],
    shortfall_losses: dict[str, list[float]],
    splits: dict[str, list[ActionRecord]],
    *,
    alpha: float,
    delta: float,
    tool_cost: float,
) -> dict[str, Any]:
    fractions = (0.02, 0.05, 0.10, 0.20, 0.30)
    per_policy_delta = delta / len(fractions)
    candidates = []
    for fraction in fractions:
        acquisition_threshold = _top_fraction_threshold(
            base_scores["acceptance"], fraction
        )
        sparse_acceptance, acquired_acceptance = _apply_sparse_acquisition(
            base_scores["acceptance"],
            verification_scores["acceptance"],
            threshold=acquisition_threshold,
        )
        risk_selection = select_risk_controlled_grid(
            sparse_acceptance,
            labels["acceptance"],
            coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
            alpha=alpha,
            delta=per_policy_delta,
        )
        acceptance = _evaluate_threshold(
            sparse_acceptance,
            labels["acceptance"],
            gains["acceptance"],
            splits["acceptance"],
            threshold=risk_selection.threshold,
            tool_cost=tool_cost,
            acquired_count=acquired_acceptance,
            shortfall_losses=shortfall_losses["acceptance"],
        )
        candidates.append(
            {
                "acquisition_fraction": fraction,
                "acquisition_threshold": acquisition_threshold,
                "risk_selection": risk_selection.to_json(),
                "acceptance": acceptance,
            }
        )
    viable = [
        candidate
        for candidate in candidates
        if candidate["risk_selection"]["selected_count"] > 0
        and candidate["acceptance"]["population_net_gain"] > 0.0
    ]
    if not viable:
        no_op_test = _evaluate_threshold(
            [0.0] * len(labels["test"]),
            labels["test"],
            gains["test"],
            splits["test"],
            threshold=1.0 + 1e-12,
            tool_cost=tool_cost,
            acquired_count=0,
            shortfall_losses=shortfall_losses["test"],
        )
        return {
            "selected_policy": "no_op",
            "alpha": alpha,
            "family_wise_delta": delta,
            "per_acquisition_policy_delta": per_policy_delta,
            "candidates": candidates,
            "test": no_op_test,
        }
    selected = max(
        viable,
        key=lambda candidate: (
            candidate["acceptance"]["population_net_gain"],
            -candidate["acquisition_fraction"],
        ),
    )
    sparse_test, acquired_test = _apply_sparse_acquisition(
        base_scores["test"],
        verification_scores["test"],
        threshold=selected["acquisition_threshold"],
    )
    test = _evaluate_threshold(
        sparse_test,
        labels["test"],
        gains["test"],
        splits["test"],
        threshold=selected["risk_selection"]["threshold"],
        tool_cost=tool_cost,
        acquired_count=acquired_test,
        shortfall_losses=shortfall_losses["test"],
    )
    return {
        "selected_policy": "certified_acquire_then_execute",
        "alpha": alpha,
        "family_wise_delta": delta,
        "per_acquisition_policy_delta": per_policy_delta,
        "selected_acquisition_fraction": selected["acquisition_fraction"],
        "selected_acquisition_threshold": selected["acquisition_threshold"],
        "selected_risk_selection": selected["risk_selection"],
        "selected_acceptance": selected["acceptance"],
        "candidates": candidates,
        "test": test,
    }


def _run_safe_activation_policy(
    sparse_scores: dict[str, list[float]],
    metadata: dict[str, Any],
    labels: dict[str, list[int]],
    gains: dict[str, list[float]],
    shortfall_losses: dict[str, list[float]],
    splits: dict[str, list[ActionRecord]],
    *,
    alpha: float,
    delta: float,
) -> dict[str, Any]:
    tool_cost = float(metadata["primary_tool_cost"])
    selection = select_risk_controlled_grid(
        sparse_scores["acceptance"],
        labels["acceptance"],
        coverages=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
        alpha=alpha,
        delta=delta,
    )
    acceptance = _evaluate_threshold(
        sparse_scores["acceptance"],
        labels["acceptance"],
        gains["acceptance"],
        splits["acceptance"],
        threshold=selection.threshold,
        tool_cost=tool_cost,
        acquired_count=int(metadata["acquisition_counts"]["acceptance"]),
        shortfall_losses=shortfall_losses["acceptance"],
    )
    active = selection.selected_count > 0 and acceptance["population_net_gain"] > 0.0
    if active:
        test = _evaluate_threshold(
            sparse_scores["test"],
            labels["test"],
            gains["test"],
            splits["test"],
            threshold=selection.threshold,
            tool_cost=tool_cost,
            acquired_count=int(metadata["acquisition_counts"]["test"]),
            shortfall_losses=shortfall_losses["test"],
        )
    else:
        test = _evaluate_threshold(
            [0.0] * len(labels["test"]),
            labels["test"],
            gains["test"],
            splits["test"],
            threshold=1.0 + 1e-12,
            tool_cost=tool_cost,
            acquired_count=0,
            shortfall_losses=shortfall_losses["test"],
        )
    return {
        "selected_policy": "activate_frozen_contract" if active else "no_op",
        "risk_selection": selection.to_json(),
        "acceptance": acceptance,
        "test": test,
    }


def _top_fraction_threshold(scores: list[float], fraction: float) -> float:
    if not scores:
        raise ValueError("scores cannot be empty")
    count = max(1, min(len(scores), math.ceil(len(scores) * fraction)))
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    next_score = scores[order[count]] if count < len(order) else -1.0
    return (scores[order[count - 1]] + next_score) / 2.0


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(clipped / (1.0 - clipped))
