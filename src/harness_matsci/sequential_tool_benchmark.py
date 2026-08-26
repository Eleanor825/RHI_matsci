from __future__ import annotations

import hashlib
import json
import math
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .schema import ActionRecord


STAGES = ("base", "screening", "verification")

_TASK_FILES = {
    "matbench_pairwise": "matbench_pairwise_actions.jsonl",
    "discover_unique": "unique_materials_actions.jsonl",
    "extreme_properties": "extreme_properties_actions.jsonl",
}

_FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "label",
        "outcome_success",
        "metric_value",
        "utility",
        "hidden_evaluation",
        "true_gap",
        "discovery_score",
        "reward",
        "five_hit",
        "all_hit",
        "exact_generated",
        "exact_target",
    }
)

_ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og".split()
)
_ATOMIC_NUMBER = {element: index + 1 for index, element in enumerate(_ELEMENTS)}
_GLOBAL_MAGPIE_CACHE: dict[str, tuple[float, ...]] = {}
_GLOBAL_RDKIT_CACHE: dict[str, dict[str, float]] = {}
_GLOBAL_MAGPIE_FEATURIZER: Any | None = None


@dataclass(frozen=True)
class ToolStageSpec:
    name: str
    cost: float
    scalar_noise: float
    fidelity: float


@dataclass(frozen=True)
class SequentialToolObservation:
    record_id: str
    benchmark: str
    stage: str
    evidence_replica: int
    tool_cost: float
    record: ActionRecord
    visible_outputs: dict[str, Any]
    provenance: dict[str, Any]
    latent_margin_for_audit_only: float

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown observation stage {self.stage!r}")
        audit_visible_payload(self.visible_payload())

    def visible_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "benchmark": self.benchmark,
            "stage": self.stage,
            "evidence_replica": self.evidence_replica,
            "tool_cost": self.tool_cost,
            "record": _visible_record_payload(self.record),
            "visible_outputs": dict(self.visible_outputs),
            "provenance": dict(self.provenance),
        }

    def to_json(self, *, include_audit_fields: bool = False) -> dict[str, Any]:
        payload = self.visible_payload()
        if include_audit_fields:
            payload["audit_only"] = {"latent_margin": self.latent_margin_for_audit_only}
        return payload


class SequentialToolEnvironment:
    """Leakage-audited, costed observations for historical science actions.

    The environment is deliberately marked semi-synthetic: task-native tool
    measurements are generated as noisy observations of historical benchmark
    measurements.  Exact outcomes remain evaluation-only.
    """

    def __init__(
        self,
        payloads_by_id: dict[str, dict[str, Any]],
        unique_discovery_threshold: float,
        stage_specs: dict[str, ToolStageSpec] | None = None,
    ) -> None:
        self.payloads_by_id = payloads_by_id
        self.unique_discovery_threshold = float(unique_discovery_threshold)
        self.stage_specs = stage_specs or {
            "base": ToolStageSpec("base", cost=0.0, scalar_noise=0.0, fidelity=0.0),
            "screening": ToolStageSpec("screening", cost=0.025, scalar_noise=0.18, fidelity=0.55),
            "verification": ToolStageSpec("verification", cost=0.075, scalar_noise=0.06, fidelity=0.90),
        }

    @classmethod
    def fit(
        cls,
        raw_data_dir: str | Path,
        train_records: Iterable[ActionRecord],
    ) -> "SequentialToolEnvironment":
        payloads = load_raw_payload_index(raw_data_dir)
        train_ids = {record.record_id for record in train_records}
        unique_scores = [
            _unique_discovery_score(payloads[record_id])
            for record_id in train_ids
            if record_id in payloads and _benchmark_for_payload(payloads[record_id]) == "discover_unique"
        ]
        if not unique_scores:
            raise ValueError("training records contain no unique-material discovery scores")
        threshold = _quantile(unique_scores, 0.90)
        return cls(payloads, threshold)

    def latent_margin(self, record: ActionRecord) -> float:
        payload = self._payload(record)
        if record.benchmark == "matbench_pairwise":
            return float(payload["tool_outputs"]["hidden_evaluation"]["true_gap"])
        if record.benchmark == "discover_unique":
            return _unique_discovery_score(payload) - self.unique_discovery_threshold
        if record.benchmark == "extreme_properties":
            error_ratios = _extreme_error_ratios(payload)
            return min(1.0 - ratio for ratio in error_ratios.values())
        raise ValueError(f"unsupported benchmark {record.benchmark!r}")

    def observe(
        self,
        record: ActionRecord,
        *,
        stage: str,
        evidence_replica: int,
        run_seed: int,
    ) -> SequentialToolObservation:
        if stage not in self.stage_specs:
            raise ValueError(f"unknown stage {stage!r}")
        spec = self.stage_specs[stage]
        latent_margin = self.latent_margin(record)
        if stage == "base":
            output_record = replace(
                record,
                features={
                    **record.features,
                    "tool_stage_screening": 0.0,
                    "tool_stage_verification": 0.0,
                    "tool_cost": 0.0,
                },
                metadata={
                    **_visible_metadata(record.metadata),
                    "tool_stage": "base",
                    "tool_observation_semantics": "none",
                },
            )
            return SequentialToolObservation(
                record_id=record.record_id,
                benchmark=record.benchmark,
                stage=stage,
                evidence_replica=evidence_replica,
                tool_cost=0.0,
                record=output_record,
                visible_outputs={"observation": "base_state_only"},
                provenance=self._provenance(spec, evidence_replica, run_seed),
                latent_margin_for_audit_only=latent_margin,
            )
        payload = self._payload(record)
        rng = random.Random(_stable_seed(run_seed, record.record_id, stage, evidence_replica))
        if record.benchmark == "matbench_pairwise":
            visible_outputs, feature_updates, evidence = self._observe_matbench(payload, spec, rng)
        elif record.benchmark == "discover_unique":
            visible_outputs, feature_updates, evidence = self._observe_unique(payload, spec, rng)
        elif record.benchmark == "extreme_properties":
            visible_outputs, feature_updates, evidence = self._observe_extreme(payload, spec, rng)
        else:
            raise ValueError(f"unsupported benchmark {record.benchmark!r}")
        output_record = replace(
            record,
            evidence=[*record.evidence, *evidence],
            features={
                **record.features,
                **feature_updates,
                "tool_stage_screening": float(stage == "screening"),
                "tool_stage_verification": float(stage == "verification"),
                "tool_cost": spec.cost,
            },
            metadata={
                **_visible_metadata(record.metadata),
                "tool_stage": stage,
                "tool_observation_semantics": "semi_synthetic_task_native_noisy_measurement",
            },
        )
        return SequentialToolObservation(
            record_id=record.record_id,
            benchmark=record.benchmark,
            stage=stage,
            evidence_replica=evidence_replica,
            tool_cost=spec.cost,
            record=output_record,
            visible_outputs=visible_outputs,
            provenance=self._provenance(spec, evidence_replica, run_seed),
            latent_margin_for_audit_only=latent_margin,
        )

    def label_consistency(self, records: Iterable[ActionRecord]) -> dict[str, float]:
        totals: dict[str, int] = {}
        matches: dict[str, int] = {}
        for record in records:
            totals[record.benchmark] = totals.get(record.benchmark, 0) + 1
            margin_label = int(self.latent_margin(record) > 0.0)
            matches[record.benchmark] = matches.get(record.benchmark, 0) + int(margin_label == record.label)
        return {task: matches.get(task, 0) / count for task, count in totals.items()}

    def _payload(self, record: ActionRecord) -> dict[str, Any]:
        try:
            payload = self.payloads_by_id[record.record_id]
        except KeyError as exc:
            raise KeyError(f"raw payload missing for {record.record_id}") from exc
        benchmark = _benchmark_for_payload(payload)
        if benchmark != record.benchmark:
            raise ValueError(f"task mismatch for {record.record_id}: {benchmark} != {record.benchmark}")
        return payload

    def _provenance(
        self,
        spec: ToolStageSpec,
        evidence_replica: int,
        run_seed: int,
    ) -> dict[str, Any]:
        return {
            "source_type": "controlled_offline_replay",
            "observation_channel": "semi_synthetic",
            "online": False,
            "matbot_runtime": False,
            "hidden_outcome_exposed": False,
            "stage_spec": asdict(spec),
            "evidence_replica": evidence_replica,
            "run_seed": run_seed,
        }

    def _observe_matbench(
        self,
        payload: dict[str, Any],
        spec: ToolStageSpec,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, float], list[str]]:
        hidden = payload["tool_outputs"]["hidden_evaluation"]
        chosen_id = str(hidden["chosen"]).lower()
        other_id = str(hidden["other"]).lower()
        chosen = float(hidden[f"candidate_{chosen_id}"]["normalized_k_vrh"])
        other = float(hidden[f"candidate_{other_id}"]["normalized_k_vrh"])
        chosen_estimate = chosen + rng.gauss(0.0, spec.scalar_noise)
        other_estimate = other + rng.gauss(0.0, spec.scalar_noise)
        gap = chosen_estimate - other_estimate
        gap_standard_error = math.sqrt(2.0) * spec.scalar_noise
        probability = _normal_cdf(gap / max(1e-9, gap_standard_error))
        outputs = {
            "estimated_chosen_bulk_modulus_percentile": _clip(chosen_estimate),
            "estimated_alternative_bulk_modulus_percentile": _clip(other_estimate),
            "estimated_pairwise_margin": gap,
            "declared_margin_standard_error": gap_standard_error,
            "probability_chosen_is_better": probability,
        }
        features = {
            "tool_estimated_margin": gap,
            "tool_positive_probability": probability,
            "tool_uncertainty": gap_standard_error,
            "tool_estimated_primary": _clip(chosen_estimate),
            "tool_estimated_secondary": _clip(other_estimate),
            "tool_fidelity": spec.fidelity,
        }
        evidence = [
            f"{spec.name} property model estimated chosen percentile={_clip(chosen_estimate):.4f}.",
            f"Estimated alternative percentile={_clip(other_estimate):.4f}; declared margin SE={gap_standard_error:.4f}.",
        ]
        return outputs, features, evidence

    def _observe_unique(
        self,
        payload: dict[str, Any],
        spec: ToolStageSpec,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, float], list[str]]:
        scores = payload["tool_outputs"]["scores"]
        performance = float(scores["performance_score"])
        uniqueness = float(scores["uniqueness_score"])
        performance_estimate = _clip(performance + rng.gauss(0.0, spec.scalar_noise))
        uniqueness_estimate = _clip(uniqueness + rng.gauss(0.0, spec.scalar_noise))
        composite = 0.5 * (performance_estimate + uniqueness_estimate)
        composite_standard_error = spec.scalar_noise / math.sqrt(2.0)
        margin = composite - self.unique_discovery_threshold
        probability = _normal_cdf(margin / max(1e-9, composite_standard_error))
        outputs = {
            "estimated_performance_percentile": performance_estimate,
            "estimated_train_library_novelty": uniqueness_estimate,
            "estimated_composite_score": composite,
            "training_only_advancement_threshold": self.unique_discovery_threshold,
            "declared_composite_standard_error": composite_standard_error,
            "probability_above_advancement_threshold": probability,
        }
        features = {
            "tool_estimated_margin": margin,
            "tool_positive_probability": probability,
            "tool_uncertainty": composite_standard_error,
            "tool_estimated_primary": performance_estimate,
            "tool_estimated_secondary": uniqueness_estimate,
            "tool_fidelity": spec.fidelity,
        }
        evidence = [
            f"{spec.name} estimated performance percentile={performance_estimate:.4f} and train-library novelty={uniqueness_estimate:.4f}.",
            f"Estimated composite={composite:.4f}; train-only threshold={self.unique_discovery_threshold:.4f}; declared SE={composite_standard_error:.4f}.",
        ]
        return outputs, features, evidence

    def _observe_extreme(
        self,
        payload: dict[str, Any],
        spec: ToolStageSpec,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, float], list[str]]:
        exact_ratios = _extreme_error_ratios(payload)
        estimated_ratios = {
            name: max(0.0, ratio + rng.gauss(0.0, spec.scalar_noise))
            for name, ratio in exact_ratios.items()
        }
        slacks = {name: 1.0 - ratio for name, ratio in estimated_ratios.items()}
        worst_slack = min(slacks.values())
        hit_probabilities = {
            name: _normal_cdf(slack / max(1e-9, spec.scalar_noise))
            for name, slack in slacks.items()
        }
        all_hit_probability = math.prod(hit_probabilities.values())
        mean_ratio = sum(estimated_ratios.values()) / len(estimated_ratios)
        outputs = {
            "estimated_normalized_property_errors": estimated_ratios,
            "estimated_worst_constraint_slack": worst_slack,
            "estimated_mean_normalized_error": mean_ratio,
            "declared_ratio_standard_error": spec.scalar_noise,
            "probability_all_constraints_hit": all_hit_probability,
        }
        features = {
            "tool_estimated_margin": worst_slack,
            "tool_positive_probability": all_hit_probability,
            "tool_uncertainty": spec.scalar_noise,
            "tool_estimated_primary": mean_ratio,
            "tool_estimated_secondary": max(estimated_ratios.values()),
            "tool_fidelity": spec.fidelity,
        }
        for name, ratio in estimated_ratios.items():
            features[f"tool_error_ratio_{name.lower()}"] = ratio
        evidence = [
            f"{spec.name} estimated normalized errors for seven target properties.",
            f"Worst estimated constraint slack={worst_slack:.4f}; probability all constraints hit={all_hit_probability:.4f}; declared ratio SE={spec.scalar_noise:.4f}.",
        ]
        return outputs, features, evidence


@dataclass(frozen=True)
class _FittedSurrogateReplica:
    vectorizer: Any
    model: Any
    target_mean: float
    target_scale: float
    residual_scale: float
    training_count: int


class TrainOnlySurrogateToolEnvironment(SequentialToolEnvironment):
    """Offline scientific tools fitted only on the training partition.

    Unlike ``SequentialToolEnvironment``, non-base observations are predictions
    of frozen train-only surrogate replicas. Hidden feedback, acceptance, and
    test outcomes are never used to create visible tool outputs.
    """

    def __init__(
        self,
        payloads_by_id: dict[str, dict[str, Any]],
        unique_discovery_threshold: float,
        train_records: tuple[ActionRecord, ...],
        train_margins: dict[str, float],
        *,
        environment_seed: int,
        stage_specs: dict[str, ToolStageSpec] | None = None,
    ) -> None:
        super().__init__(payloads_by_id, unique_discovery_threshold, stage_specs)
        self.train_records = train_records
        self.train_margins = dict(train_margins)
        self.environment_seed = int(environment_seed)
        self.training_hash = hashlib.sha256(
            "\n".join(sorted(record.record_id for record in train_records)).encode()
        ).hexdigest()
        self._surrogate_cache: dict[tuple[str, str, int], _FittedSurrogateReplica] = {}
        self._surrogate_design_cache: dict[
            tuple[str, str], tuple[Any, Any, list[ActionRecord], list[float], float, float]
        ] = {}

    @classmethod
    def fit(
        cls,
        raw_data_dir: str | Path,
        train_records: Iterable[ActionRecord],
        *,
        environment_seed: int,
    ) -> "TrainOnlySurrogateToolEnvironment":
        records = tuple(train_records)
        base = SequentialToolEnvironment.fit(raw_data_dir, records)
        train_margins = {record.record_id: base.latent_margin(record) for record in records}
        return cls(
            base.payloads_by_id,
            base.unique_discovery_threshold,
            records,
            train_margins,
            environment_seed=environment_seed,
            stage_specs=base.stage_specs,
        )

    def observe(
        self,
        record: ActionRecord,
        *,
        stage: str,
        evidence_replica: int,
        run_seed: int,
    ) -> SequentialToolObservation:
        if stage == "base":
            return super().observe(
                record,
                stage=stage,
                evidence_replica=evidence_replica,
                run_seed=run_seed,
            )
        if stage not in self.stage_specs:
            raise ValueError(f"unknown stage {stage!r}")
        spec = self.stage_specs[stage]
        predicted_margin, residual_scale, surrogate_family, scientific_outputs = self._predict_margin(
            record,
            stage=stage,
            evidence_replica=evidence_replica,
        )
        positive_probability = _normal_cdf(
            predicted_margin / max(1e-9, residual_scale)
        )
        visible_outputs = {
            "predicted_task_native_margin": predicted_margin,
            "train_residual_scale": residual_scale,
            "probability_positive_margin": positive_probability,
            "surrogate_family": surrogate_family,
            **scientific_outputs,
        }
        evidence = [
            f"{spec.name} train-only surrogate predicted margin={predicted_margin:.4f}.",
            f"Training residual scale={residual_scale:.4f}; positive-margin probability={positive_probability:.4f}.",
        ]
        output_record = replace(
            record,
            evidence=[*record.evidence, *evidence],
            features={
                **record.features,
                "tool_estimated_margin": predicted_margin,
                "tool_positive_probability": positive_probability,
                "tool_uncertainty": residual_scale,
                "tool_fidelity": spec.fidelity,
                "tool_stage_screening": float(stage == "screening"),
                "tool_stage_verification": float(stage == "verification"),
                "tool_cost": spec.cost,
            },
            metadata={
                **_visible_metadata(record.metadata),
                "tool_stage": stage,
                "tool_observation_semantics": "train_only_surrogate_prediction",
                "tool_training_hash": self.training_hash,
            },
        )
        return SequentialToolObservation(
            record_id=record.record_id,
            benchmark=record.benchmark,
            stage=stage,
            evidence_replica=evidence_replica,
            tool_cost=spec.cost,
            record=output_record,
            visible_outputs=visible_outputs,
            provenance=self._provenance(spec, evidence_replica, run_seed),
            latent_margin_for_audit_only=self.latent_margin(record),
        )

    def _predict_margin(
        self,
        record: ActionRecord,
        *,
        stage: str,
        evidence_replica: int,
    ) -> tuple[float, float, str, dict[str, Any]]:
        replica = self._surrogate(stage, record.benchmark, evidence_replica)
        matrix = replica.vectorizer.transform([_surrogate_text(record)])
        normalized_prediction = float(replica.model.predict(matrix)[0])
        predicted_margin = replica.target_mean + replica.target_scale * normalized_prediction
        return predicted_margin, replica.residual_scale, "hashed_character_ngram_ridge", {}

    def _provenance(
        self,
        spec: ToolStageSpec,
        evidence_replica: int,
        run_seed: int,
    ) -> dict[str, Any]:
        return {
            "source_type": "fitted_offline_scientific_surrogate",
            "observation_channel": "train_only_surrogate_prediction",
            "online": False,
            "matbot_runtime": False,
            "hidden_outcome_exposed": False,
            "nontraining_outcome_used_for_observation": False,
            "training_hash": self.training_hash,
            "training_count": len(self.train_records),
            "stage_spec": asdict(spec),
            "evidence_replica": evidence_replica,
            "environment_seed": self.environment_seed,
            "run_seed": run_seed,
        }

    def audit_nontraining_outcome_invariance(
        self,
        records: Iterable[ActionRecord],
        *,
        evidence_replica: int,
        run_seed: int,
    ) -> dict[str, Any]:
        audited = 0
        task_passes: dict[str, bool] = {}
        for task in sorted({record.benchmark for record in records}):
            record = next(record for record in records if record.benchmark == task)
            original_payload = deepcopy(self.payloads_by_id[record.record_id])
            try:
                baseline = self.observe(
                    record,
                    stage="verification",
                    evidence_replica=evidence_replica,
                    run_seed=run_seed,
                ).visible_payload()
                _corrupt_hidden_outcome(self.payloads_by_id[record.record_id], task)
                corrupted_record = replace(record, label=1 - record.label)
                corrupted = self.observe(
                    corrupted_record,
                    stage="verification",
                    evidence_replica=evidence_replica,
                    run_seed=run_seed,
                ).visible_payload()
                task_passes[task] = json.dumps(
                    baseline, sort_keys=True, ensure_ascii=False
                ) == json.dumps(corrupted, sort_keys=True, ensure_ascii=False)
                audited += 1
            finally:
                self.payloads_by_id[record.record_id] = original_payload
        return {
            "passed": bool(task_passes) and all(task_passes.values()),
            "audited_records": audited,
            "task_passes": task_passes,
            "mutations": "nontraining labels and hidden task-native margins",
        }

    def _surrogate(
        self,
        stage: str,
        benchmark: str,
        evidence_replica: int,
    ) -> _FittedSurrogateReplica:
        key = (stage, benchmark, evidence_replica)
        if key in self._surrogate_cache:
            return self._surrogate_cache[key]
        from sklearn.linear_model import Ridge

        vectorizer, matrix, records, targets, target_mean, target_scale = (
            self._surrogate_design(stage, benchmark)
        )
        normalized_targets = [(target - target_mean) / target_scale for target in targets]
        ridge_alpha = 12.0 if stage == "screening" else 3.0
        rng = random.Random(
            _stable_seed(self.environment_seed, benchmark, stage, evidence_replica)
        )
        sample_weights = [0.0] * len(records)
        for _ in records:
            sample_weights[rng.randrange(len(records))] += 1.0
        model = Ridge(alpha=ridge_alpha).fit(
            matrix,
            normalized_targets,
            sample_weight=sample_weights,
        )
        training_predictions = model.predict(matrix)
        residual_scale = math.sqrt(
            sum(
                (
                    target
                    - (target_mean + target_scale * float(prediction))
                )
                ** 2
                for target, prediction in zip(targets, training_predictions)
            )
            / len(targets)
        ) or 1e-6
        fitted = _FittedSurrogateReplica(
            vectorizer=vectorizer,
            model=model,
            target_mean=target_mean,
            target_scale=target_scale,
            residual_scale=residual_scale,
            training_count=len(records),
        )
        self._surrogate_cache[key] = fitted
        return fitted

    def _surrogate_design(
        self,
        stage: str,
        benchmark: str,
    ) -> tuple[Any, Any, list[ActionRecord], list[float], float, float]:
        key = (stage, benchmark)
        if key in self._surrogate_design_cache:
            return self._surrogate_design_cache[key]
        from sklearn.feature_extraction.text import HashingVectorizer

        records = [record for record in self.train_records if record.benchmark == benchmark]
        if not records:
            raise ValueError(f"no training records for surrogate task {benchmark}")
        targets = [self.train_margins[record.record_id] for record in records]
        target_mean = sum(targets) / len(targets)
        target_scale = math.sqrt(
            sum((target - target_mean) ** 2 for target in targets) / len(targets)
        ) or 1.0
        if stage == "screening":
            vectorizer = HashingVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                n_features=1024,
                alternate_sign=False,
                norm="l2",
            )
        else:
            vectorizer = HashingVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                n_features=4096,
                alternate_sign=False,
                norm="l2",
            )
        matrix = vectorizer.transform([_surrogate_text(record) for record in records])
        design = (vectorizer, matrix, records, targets, target_mean, target_scale)
        self._surrogate_design_cache[key] = design
        return design


@dataclass(frozen=True)
class _FittedStructuredReplica:
    feature_names: tuple[str, ...]
    model: Any
    margin_scale: float
    residual_scale: float


class TrainOnlyStructuredSurrogateToolEnvironment(TrainOnlySurrogateToolEnvironment):
    """Train-only task-specific tools using structured scientific descriptors."""

    def __init__(
        self,
        payloads_by_id: dict[str, dict[str, Any]],
        unique_discovery_threshold: float,
        train_records: tuple[ActionRecord, ...],
        train_margins: dict[str, float],
        *,
        environment_seed: int,
        stage_specs: dict[str, ToolStageSpec] | None = None,
    ) -> None:
        super().__init__(
            payloads_by_id,
            unique_discovery_threshold,
            train_records,
            train_margins,
            environment_seed=environment_seed,
            stage_specs=stage_specs,
        )
        self._structured_cache: dict[
            tuple[str, str, int], _FittedStructuredReplica
        ] = {}
        self._structured_design_cache: dict[
            tuple[str, str], tuple[tuple[str, ...], Any, list[float], float]
        ] = {}

    def _predict_margin(
        self,
        record: ActionRecord,
        *,
        stage: str,
        evidence_replica: int,
    ) -> tuple[float, float, str, dict[str, Any]]:
        replica = self._structured_surrogate(stage, record.benchmark, evidence_replica)
        features = _structured_surrogate_features(record, stage)
        row = [[float(features.get(name, 0.0)) for name in replica.feature_names]]
        transformed = float(replica.model.predict(row)[0])
        transformed = min(6.0, max(-6.0, transformed))
        predicted_margin = replica.margin_scale * math.sinh(transformed)
        return predicted_margin, replica.residual_scale, "task_structured_extratrees", {}

    def _structured_surrogate(
        self,
        stage: str,
        benchmark: str,
        evidence_replica: int,
    ) -> _FittedStructuredReplica:
        key = (stage, benchmark, evidence_replica)
        if key in self._structured_cache:
            return self._structured_cache[key]
        from sklearn.ensemble import ExtraTreesRegressor

        feature_names, matrix, targets, margin_scale = self._structured_design(
            stage, benchmark
        )
        transformed_targets = [math.asinh(target / margin_scale) for target in targets]
        rng = random.Random(
            _stable_seed(self.environment_seed, benchmark, stage, evidence_replica)
        )
        sample_weights = [0.0] * len(targets)
        for _ in targets:
            sample_weights[rng.randrange(len(targets))] += 1.0
        model = ExtraTreesRegressor(
            n_estimators=64,
            max_depth=10 if stage == "screening" else None,
            min_samples_leaf=10 if stage == "screening" else 4,
            max_features=0.7 if stage == "screening" else 1.0,
            random_state=_stable_seed(
                self.environment_seed + 17,
                benchmark,
                stage,
                evidence_replica,
            )
            % (2**32 - 1),
            n_jobs=1,
        ).fit(matrix, transformed_targets, sample_weight=sample_weights)
        predictions = model.predict(matrix)
        original_predictions = [
            margin_scale * math.sinh(min(6.0, max(-6.0, float(value))))
            for value in predictions
        ]
        residual_scale = math.sqrt(
            sum((target - prediction) ** 2 for target, prediction in zip(targets, original_predictions))
            / len(targets)
        ) or 1e-6
        fitted = _FittedStructuredReplica(
            feature_names=feature_names,
            model=model,
            margin_scale=margin_scale,
            residual_scale=residual_scale,
        )
        self._structured_cache[key] = fitted
        return fitted

    def _structured_design(
        self,
        stage: str,
        benchmark: str,
    ) -> tuple[tuple[str, ...], Any, list[float], float]:
        key = (stage, benchmark)
        if key in self._structured_design_cache:
            return self._structured_design_cache[key]
        import numpy as np

        records = [record for record in self.train_records if record.benchmark == benchmark]
        targets = [self.train_margins[record.record_id] for record in records]
        feature_dicts = [_structured_surrogate_features(record, stage) for record in records]
        feature_names = tuple(sorted({name for values in feature_dicts for name in values}))
        matrix = np.asarray(
            [[float(values.get(name, 0.0)) for name in feature_names] for values in feature_dicts],
            dtype=float,
        )
        absolute = sorted(abs(target) for target in targets)
        margin_scale = _quantile(absolute, 0.75) or 1.0
        design = (feature_names, matrix, targets, margin_scale)
        self._structured_design_cache[key] = design
        return design


class TrainOnlyStructuredDirectToolEnvironment(TrainOnlyStructuredSurrogateToolEnvironment):
    """Structured train-only tools with deterministic descriptors at base."""

    def observe(
        self,
        record: ActionRecord,
        *,
        stage: str,
        evidence_replica: int,
        run_seed: int,
    ) -> SequentialToolObservation:
        observation = super().observe(
            record,
            stage=stage,
            evidence_replica=evidence_replica,
            run_seed=run_seed,
        )
        if stage != "base":
            return observation
        descriptor_values = {
            f"deterministic::{name}": value
            for name, value in _structured_surrogate_features(record, "screening").items()
            if not name.startswith("base::")
        }
        enriched_record = replace(
            observation.record,
            features={**observation.record.features, **descriptor_values},
            metadata={
                **observation.record.metadata,
                "tool_observation_semantics": "base_state_with_deterministic_scientific_descriptors",
            },
        )
        return SequentialToolObservation(
            record_id=observation.record_id,
            benchmark=observation.benchmark,
            stage=observation.stage,
            evidence_replica=observation.evidence_replica,
            tool_cost=observation.tool_cost,
            record=enriched_record,
            visible_outputs={
                **observation.visible_outputs,
                "deterministic_descriptor_count": len(descriptor_values),
            },
            provenance={
                **observation.provenance,
                "deterministic_base_descriptors": True,
            },
            latent_margin_for_audit_only=observation.latent_margin_for_audit_only,
        )


@dataclass(frozen=True)
class _RealToolRegressor:
    model: Any
    residual_scale: float


class RealScientificToolEnvironment(TrainOnlyStructuredDirectToolEnvironment):
    """Offline trajectories generated by Matminer and RDKit scientific tools."""

    def __init__(
        self,
        payloads_by_id: dict[str, dict[str, Any]],
        unique_discovery_threshold: float,
        train_records: tuple[ActionRecord, ...],
        train_margins: dict[str, float],
        *,
        environment_seed: int,
        stage_specs: dict[str, ToolStageSpec] | None = None,
    ) -> None:
        super().__init__(
            payloads_by_id,
            unique_discovery_threshold,
            train_records,
            train_margins,
            environment_seed=environment_seed,
            stage_specs=stage_specs,
        )
        self._real_model_cache: dict[tuple[str, str, int], _RealToolRegressor] = {}
        self._real_design_cache: dict[tuple[str, str], tuple[Any, list[float]]] = {}
        self._novelty_cache: dict[str, tuple[Any, Any, list[float]]] = {}

    def _predict_margin(
        self,
        record: ActionRecord,
        *,
        stage: str,
        evidence_replica: int,
    ) -> tuple[float, float, str, dict[str, Any]]:
        if record.benchmark == "matbench_pairwise":
            return self._predict_pairwise(record, stage, evidence_replica)
        if record.benchmark == "discover_unique":
            return self._predict_unique(record, stage, evidence_replica)
        if record.benchmark == "extreme_properties":
            return self._predict_extreme(record, stage, evidence_replica)
        raise ValueError(f"unsupported benchmark {record.benchmark!r}")

    def _predict_pairwise(
        self,
        record: ActionRecord,
        stage: str,
        evidence_replica: int,
    ) -> tuple[float, float, str, dict[str, Any]]:
        candidates = _pairwise_visible_candidates(record)
        chosen = "A" if "candidate A" in record.candidate_action else "B"
        other = "B" if chosen == "A" else "A"
        regressor = self._real_regressor("matbench_pairwise", stage, evidence_replica)
        chosen_value = float(
            regressor.model.predict([_magpie_material_vector(*candidates[chosen], stage)])[0]
        )
        other_value = float(
            regressor.model.predict([_magpie_material_vector(*candidates[other], stage)])[0]
        )
        margin = chosen_value - other_value
        residual = math.sqrt(2.0) * regressor.residual_scale
        return margin, residual, "matminer_magpie_bulk_modulus", {
            "predicted_chosen_normalized_bulk_modulus": chosen_value,
            "predicted_alternative_normalized_bulk_modulus": other_value,
            "chosen_composition": candidates[chosen][0],
            "alternative_composition": candidates[other][0],
        }

    def _predict_unique(
        self,
        record: ActionRecord,
        stage: str,
        evidence_replica: int,
    ) -> tuple[float, float, str, dict[str, Any]]:
        formula, crystal, space_group = _unique_visible_candidate(record)
        regressor = self._real_regressor("discover_unique", stage, evidence_replica)
        vector = _magpie_material_vector(formula, crystal, space_group, stage)
        performance = _clip(float(regressor.model.predict([vector])[0]))
        novelty = self._novelty_percentile(formula, crystal, space_group, stage)
        discovery = 0.5 * (performance + novelty)
        margin = discovery - self.unique_discovery_threshold
        return margin, max(1e-6, 0.5 * regressor.residual_scale), "matminer_magpie_discover", {
            "predicted_performance_percentile": performance,
            "train_library_novelty_percentile": novelty,
            "predicted_discovery_score": discovery,
            "training_only_advancement_threshold": self.unique_discovery_threshold,
            "composition": formula,
        }

    def _predict_extreme(
        self,
        record: ActionRecord,
        stage: str,
        evidence_replica: int,
    ) -> tuple[float, float, str, dict[str, Any]]:
        payload = self._payload(record)
        target = payload["tool_outputs"]["target"]
        bounds = payload["source"]["target_bounds"]
        smiles = record.candidate_action.split(":", 1)[-1].strip()
        try:
            generated = _rdkit_property_values(smiles)
        except ValueError:
            return -1.0, 1.0, "rdkit_parse_failure", {
                "rdkit_parse_success": False,
                "rdkit_generated_properties": {},
                "predicted_drd2": None,
                "retrieved_target_profile": {
                    name: float(target[name]) for name in bounds
                },
                "normalized_property_errors": {},
                "target_id": str(target["target_id"]),
            }
        regressor = self._real_regressor("extreme_properties", stage, evidence_replica)
        drd2 = _clip(float(regressor.model.predict([_rdkit_model_vector(smiles, stage)])[0]))
        generated_with_drd2 = {**generated, "DRD2": drd2}
        normalized_errors = {
            name: abs(float(generated_with_drd2[name]) - float(target[name]))
            / max(1e-12, float(bounds[name]))
            for name in bounds
        }
        margin = min(1.0 - error for error in normalized_errors.values())
        residual = regressor.residual_scale / max(1e-12, float(bounds["DRD2"]))
        return margin, max(1e-6, residual), "rdkit_descriptors_plus_train_drd2", {
            "rdkit_generated_properties": generated,
            "predicted_drd2": drd2,
            "retrieved_target_profile": {
                name: float(target[name]) for name in bounds
            },
            "normalized_property_errors": normalized_errors,
            "target_id": str(target["target_id"]),
        }

    def _real_regressor(
        self,
        benchmark: str,
        stage: str,
        evidence_replica: int,
    ) -> _RealToolRegressor:
        key = (benchmark, stage, evidence_replica)
        if key in self._real_model_cache:
            return self._real_model_cache[key]
        from sklearn.ensemble import ExtraTreesRegressor

        matrix, targets = self._real_design(benchmark, stage)
        rng = random.Random(
            _stable_seed(self.environment_seed, benchmark, stage, evidence_replica)
        )
        sample_weights = [0.0] * len(targets)
        for _ in targets:
            sample_weights[rng.randrange(len(targets))] += 1.0
        model = ExtraTreesRegressor(
            n_estimators=48 if stage == "screening" else 96,
            max_depth=12 if stage == "screening" else None,
            min_samples_leaf=8 if stage == "screening" else 3,
            max_features=0.65 if stage == "screening" else 1.0,
            random_state=_stable_seed(
                self.environment_seed + 31,
                benchmark,
                stage,
                evidence_replica,
            )
            % (2**32 - 1),
            n_jobs=1,
        ).fit(matrix, targets, sample_weight=sample_weights)
        predictions = model.predict(matrix)
        residual = math.sqrt(
            sum((target - float(prediction)) ** 2 for target, prediction in zip(targets, predictions))
            / len(targets)
        ) or 1e-6
        fitted = _RealToolRegressor(model=model, residual_scale=residual)
        self._real_model_cache[key] = fitted
        return fitted

    def _real_design(self, benchmark: str, stage: str) -> tuple[Any, list[float]]:
        key = (benchmark, stage)
        if key in self._real_design_cache:
            return self._real_design_cache[key]
        import numpy as np

        rows: list[list[float]] = []
        targets: list[float] = []
        for record in self.train_records:
            if record.benchmark != benchmark:
                continue
            payload = self._payload(record)
            if benchmark == "matbench_pairwise":
                hidden = payload["tool_outputs"]["hidden_evaluation"]
                for candidate_key in ("candidate_a", "candidate_b"):
                    candidate = hidden[candidate_key]
                    rows.append(
                        _magpie_material_vector(
                            str(candidate["composition"]),
                            str(candidate["crys_sys"]),
                            int(candidate["spg_num"]),
                            stage,
                        )
                    )
                    targets.append(float(candidate["normalized_k_vrh"]))
            elif benchmark == "discover_unique":
                material = payload["tool_outputs"]["matbench"]
                rows.append(
                    _magpie_material_vector(
                        str(material["composition"]),
                        str(material["crys_sys"]),
                        int(material["spg_num"]),
                        stage,
                    )
                )
                targets.append(float(payload["tool_outputs"]["scores"]["performance_score"]))
            else:
                smiles = str(payload["tool_outputs"]["generated"]["SMILES"])
                try:
                    vector = _rdkit_model_vector(smiles, stage)
                except ValueError:
                    continue
                rows.append(vector)
                targets.append(float(payload["tool_outputs"]["generated"]["DRD2"]))
        if not rows:
            raise ValueError(f"no valid training rows for real tool task {benchmark}")
        design = (np.asarray(rows, dtype=float), targets)
        self._real_design_cache[key] = design
        return design

    def _novelty_percentile(
        self,
        formula: str,
        crystal: str,
        space_group: int,
        stage: str,
    ) -> float:
        if stage not in self._novelty_cache:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            rows = []
            for record in self.train_records:
                if record.benchmark != "discover_unique":
                    continue
                material = self._payload(record)["tool_outputs"]["matbench"]
                rows.append(
                    _magpie_material_vector(
                        str(material["composition"]),
                        str(material["crys_sys"]),
                        int(material["spg_num"]),
                        stage,
                    )
                )
            matrix = np.asarray(rows, dtype=float)
            scaler = StandardScaler().fit(matrix)
            standardized = scaler.transform(matrix)
            neighbors = NearestNeighbors(n_neighbors=2).fit(standardized)
            distances = sorted(float(value[1]) for value in neighbors.kneighbors(standardized)[0])
            self._novelty_cache[stage] = (scaler, neighbors, distances)
        scaler, neighbors, distances = self._novelty_cache[stage]
        vector = [_magpie_material_vector(formula, crystal, space_group, stage)]
        distance = float(neighbors.kneighbors(scaler.transform(vector), n_neighbors=1)[0][0][0])
        return sum(value <= distance for value in distances) / len(distances)

    def _provenance(
        self,
        spec: ToolStageSpec,
        evidence_replica: int,
        run_seed: int,
    ) -> dict[str, Any]:
        from importlib.metadata import version

        return {
            **super()._provenance(spec, evidence_replica, run_seed),
            "source_type": "reproducible_domain_scientific_tools",
            "observation_channel": "matminer_rdkit_train_only_models",
            "tool_versions": {
                "matminer": version("matminer"),
                "pymatgen": version("pymatgen"),
                "rdkit": version("rdkit"),
            },
        }

def load_raw_payload_index(raw_data_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(raw_data_dir)
    payloads: dict[str, dict[str, Any]] = {}
    for filename in _TASK_FILES.values():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                record_id = str(payload["record_id"])
                if record_id in payloads:
                    raise ValueError(f"duplicate raw record_id {record_id} at {path}:{line_number}")
                payloads[record_id] = payload
    return payloads


def audit_visible_payload(payload: dict[str, Any]) -> None:
    violations: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).lower()
                if normalized in _FORBIDDEN_VISIBLE_KEYS:
                    violations.append(f"{path}.{key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
        elif isinstance(value, str):
            lowered = value.lower()
            for token in ("hidden outcome", "exact target-hit", "exact discovery score", "true gap="):
                if token in lowered:
                    violations.append(f"{path}:contains:{token}")

    visit(payload, "visible")
    if violations:
        raise ValueError(f"visible observation leaks evaluation fields: {sorted(set(violations))}")


def _visible_record_payload(record: ActionRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "benchmark": record.benchmark,
        "split": record.split,
        "visible_context": record.visible_context,
        "candidate_action": record.candidate_action,
        "action_type": record.action_type,
        "evidence": list(record.evidence),
        "features": dict(record.features),
        "metadata": _visible_metadata(record.metadata),
    }


def _visible_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if str(key).lower() not in _FORBIDDEN_VISIBLE_KEYS
    }


def _benchmark_for_payload(payload: dict[str, Any]) -> str:
    task = str(payload.get("task", ""))
    if task == "materials_pairwise_preference":
        return "matbench_pairwise"
    if task == "unique_materials_screening":
        return "discover_unique"
    if task == "extreme_property_discovery":
        return "extreme_properties"
    raise ValueError(f"unsupported raw task {task!r}")


def _unique_discovery_score(payload: dict[str, Any]) -> float:
    return float(payload["tool_outputs"]["scores"]["discovery_score"])


def _extreme_error_ratios(payload: dict[str, Any]) -> dict[str, float]:
    errors = payload["tool_outputs"]["errors"]
    bounds = payload["source"]["target_bounds"]
    return {name: float(errors[name]) / max(1e-12, float(bounds[name])) for name in sorted(bounds)}


def _stable_seed(run_seed: int, record_id: str, stage: str, evidence_replica: int) -> int:
    digest = hashlib.sha256(f"{run_seed}|{record_id}|{stage}|{evidence_replica}".encode()).hexdigest()
    return int(digest[:16], 16)


def _surrogate_text(record: ActionRecord) -> str:
    safe_metadata = _visible_metadata(record.metadata)
    payload = {
        "visible_context": record.visible_context,
        "candidate_action": record.candidate_action,
        "action_type": record.action_type,
        "evidence": list(record.evidence),
        "features": dict(sorted(record.features.items())),
        "metadata": dict(sorted(safe_metadata.items())),
    }
    audit_visible_payload(payload)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _structured_surrogate_features(record: ActionRecord, stage: str) -> dict[str, float]:
    features = {f"base::{name}": float(value) for name, value in record.features.items()}
    if record.benchmark == "matbench_pairwise":
        matches = re.findall(
            r"Candidate ([AB]): Composition: (.*?); crystal system: ([^;]+); space group: (\d+)",
            record.visible_context,
        )
        candidates = {
            label: (formula, crystal.strip().lower(), int(space_group))
            for label, formula, crystal, space_group in matches
        }
        chosen = "A" if "candidate A" in record.candidate_action else "B"
        other = "B" if chosen == "A" else "A"
        if chosen in candidates and other in candidates:
            _add_composition_features(features, "chosen", candidates[chosen][0], stage)
            _add_composition_features(features, "other", candidates[other][0], stage)
            chosen_counts = _composition_counts(candidates[chosen][0])
            other_counts = _composition_counts(candidates[other][0])
            if stage == "verification":
                for element in _ELEMENTS:
                    features[f"composition_diff::{element}"] = _fraction(
                        chosen_counts, element
                    ) - _fraction(other_counts, element)
            features["chosen_space_group"] = candidates[chosen][2] / 230.0
            features["other_space_group"] = candidates[other][2] / 230.0
            features["same_crystal_system"] = float(
                candidates[chosen][1] == candidates[other][1]
            )
    elif record.benchmark == "discover_unique":
        match = re.search(
            r"Composition: (.*?); crystal system: ([^;]+); space group: (\d+)",
            record.visible_context,
        )
        if match:
            _add_composition_features(features, "candidate", match.group(1), stage)
            features["space_group"] = int(match.group(3)) / 230.0
    elif record.benchmark == "extreme_properties":
        smiles = record.candidate_action.split(":", 1)[-1].strip()
        _add_smiles_features(features, smiles, stage)
        target = re.search(r"target (C\d+)", record.visible_context)
        if target:
            features[f"target::{target.group(1)}"] = 1.0
    return features


def _composition_counts(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for element, amount in re.findall(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)", formula):
        if element not in _ATOMIC_NUMBER:
            continue
        counts[element] = counts.get(element, 0.0) + (float(amount) if amount else 1.0)
    return counts


def _fraction(counts: dict[str, float], element: str) -> float:
    total = sum(counts.values()) or 1.0
    return counts.get(element, 0.0) / total


def _add_composition_features(
    features: dict[str, float], prefix: str, formula: str, stage: str
) -> None:
    counts = _composition_counts(formula)
    total = sum(counts.values()) or 1.0
    atomic_numbers = [
        _ATOMIC_NUMBER[element]
        for element, amount in counts.items()
        for _ in range(max(1, int(round(amount))))
    ]
    features[f"{prefix}::element_count"] = float(len(counts))
    features[f"{prefix}::atom_count"] = total
    if atomic_numbers:
        average = sum(atomic_numbers) / len(atomic_numbers)
        features[f"{prefix}::atomic_number_mean"] = average / 118.0
        features[f"{prefix}::atomic_number_min"] = min(atomic_numbers) / 118.0
        features[f"{prefix}::atomic_number_max"] = max(atomic_numbers) / 118.0
        features[f"{prefix}::atomic_number_std"] = math.sqrt(
            sum((value - average) ** 2 for value in atomic_numbers) / len(atomic_numbers)
        ) / 118.0
    entropy = -sum(
        fraction * math.log(max(fraction, 1e-12))
        for fraction in (_fraction(counts, element) for element in counts)
    )
    features[f"{prefix}::composition_entropy"] = entropy
    if stage == "verification":
        for element in _ELEMENTS:
            features[f"{prefix}::element_fraction::{element}"] = _fraction(counts, element)


def _add_smiles_features(features: dict[str, float], smiles: str, stage: str) -> None:
    tokens = re.findall(r"Cl|Br|[A-Z][a-z]?|[cnosp]", smiles)
    token_count = len(tokens) or 1
    features["smiles::length"] = len(smiles) / 500.0
    features["smiles::atom_tokens"] = token_count / 150.0
    features["smiles::rings"] = sum(character.isdigit() for character in smiles) / token_count
    features["smiles::branches"] = smiles.count("(") / token_count
    features["smiles::double_bonds"] = smiles.count("=") / token_count
    features["smiles::triple_bonds"] = smiles.count("#") / token_count
    features["smiles::charges"] = (smiles.count("+") + smiles.count("-")) / token_count
    features["smiles::aromatic"] = sum(token in "cnosp" for token in tokens) / token_count
    heteroatoms = sum(token not in {"C", "c", "H"} for token in tokens)
    features["smiles::heteroatom_ratio"] = heteroatoms / token_count
    if stage == "verification":
        for token in ("C", "c", "N", "n", "O", "o", "S", "s", "P", "p", "F", "Cl", "Br", "I", "B", "Si"):
            features[f"smiles::token_fraction::{token}"] = tokens.count(token) / token_count


def _pairwise_visible_candidates(
    record: ActionRecord,
) -> dict[str, tuple[str, str, int]]:
    matches = re.findall(
        r"Candidate ([AB]): Composition: (.*?); crystal system: ([^;]+); space group: (\d+)",
        record.visible_context,
    )
    candidates = {
        label: (formula.strip(), crystal.strip().lower(), int(space_group))
        for label, formula, crystal, space_group in matches
    }
    if set(candidates) != {"A", "B"}:
        raise ValueError(f"could not parse pairwise candidates for {record.record_id}")
    return candidates


def _unique_visible_candidate(record: ActionRecord) -> tuple[str, str, int]:
    match = re.search(
        r"Composition: (.*?); crystal system: ([^;]+); space group: (\d+)",
        record.visible_context,
    )
    if match is None:
        raise ValueError(f"could not parse unique candidate for {record.record_id}")
    return match.group(1).strip(), match.group(2).strip().lower(), int(match.group(3))


def _magpie_material_vector(
    formula: str,
    crystal: str,
    space_group: int,
    stage: str,
) -> list[float]:
    global _GLOBAL_MAGPIE_FEATURIZER
    if formula not in _GLOBAL_MAGPIE_CACHE:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition

        if _GLOBAL_MAGPIE_FEATURIZER is None:
            _GLOBAL_MAGPIE_FEATURIZER = ElementProperty.from_preset("magpie")
        _GLOBAL_MAGPIE_CACHE[formula] = tuple(
            float(value)
            for value in _GLOBAL_MAGPIE_FEATURIZER.featurize(
                Composition(formula.replace(" ", ""))
            )
        )
    magpie = list(_GLOBAL_MAGPIE_CACHE[formula])
    if stage == "screening":
        magpie = magpie[:48]
    crystal_types = (
        "cubic",
        "hexagonal",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "triclinic",
        "trigonal",
    )
    return [
        *magpie,
        space_group / 230.0,
        *(float(crystal == name) for name in crystal_types),
    ]


def _rdkit_property_values(smiles: str) -> dict[str, float]:
    if smiles in _GLOBAL_RDKIT_CACHE:
        return dict(_GLOBAL_RDKIT_CACHE[smiles])
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED

    RDLogger.DisableLog("rdApp.error")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse candidate SMILES")
    values = {
        "MW": float(Descriptors.MolWt(molecule)),
        "logP": float(Crippen.MolLogP(molecule)),
        "TPSA": float(Descriptors.TPSA(molecule)),
        "QED": float(QED.qed(molecule)),
        "HBA": float(Lipinski.NumHAcceptors(molecule)),
        "HBD": float(Lipinski.NumHDonors(molecule)),
    }
    _GLOBAL_RDKIT_CACHE[smiles] = values
    return dict(values)


def _rdkit_model_vector(smiles: str, stage: str) -> list[float]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    values = _rdkit_property_values(smiles)
    base = [values[name] for name in ("MW", "logP", "TPSA", "QED", "HBA", "HBD")]
    if stage == "screening":
        return base
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse candidate SMILES")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=256
    ).GetFingerprintAsNumPy(molecule)
    return [*base, *(float(value) for value in fingerprint)]


def _corrupt_hidden_outcome(payload: dict[str, Any], benchmark: str) -> None:
    if benchmark == "matbench_pairwise":
        hidden = payload["tool_outputs"]["hidden_evaluation"]
        hidden["true_gap"] = float(hidden["true_gap"]) + 1000.0
        for candidate in ("candidate_a", "candidate_b"):
            if candidate in hidden and "normalized_k_vrh" in hidden[candidate]:
                hidden[candidate]["normalized_k_vrh"] = 1.0 - float(
                    hidden[candidate]["normalized_k_vrh"]
                )
        return
    if benchmark == "discover_unique":
        scores = payload["tool_outputs"]["scores"]
        for key in tuple(scores):
            scores[key] = -1000.0
        return
    if benchmark == "extreme_properties":
        errors = payload["tool_outputs"]["errors"]
        for key in tuple(errors):
            errors[key] = float(errors[key]) + 1000.0
        return
    raise ValueError(f"unsupported benchmark {benchmark!r}")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    ordered = sorted(float(value) for value in values)
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
