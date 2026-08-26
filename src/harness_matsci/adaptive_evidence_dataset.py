from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .adaptive_factorized_model import FactorizedGainTrainingRow


STAGES = ("base", "screening", "verification")


@dataclass(frozen=True)
class AdaptiveEvidenceExample:
    trajectory: Mapping[str, object]

    @property
    def record_id(self) -> str:
        return str(self.trajectory["record_id"])

    @property
    def benchmark(self) -> str:
        return str(self.trajectory["benchmark"])

    @property
    def split(self) -> str:
        return str(self.trajectory["split"])

    @property
    def base_record(self) -> Mapping[str, object]:
        return self._stage("base")["visible_observation"]["record"]

    @property
    def group_id(self) -> str:
        metadata = self.base_record.get("metadata", {})
        return str(metadata.get("group_id", self.record_id))

    @property
    def normalized_action_cost(self) -> float:
        features = self.base_record.get("features", {})
        return _unit(features.get("cost", 0.5))

    @property
    def failure_harm(self) -> float:
        features = self.base_record.get("features", {})
        irreversibility = 1.0 - _unit(features.get("reversibility", 0.5))
        return 0.5 * self.normalized_action_cost + 0.5 * irreversibility

    def cumulative_evidence_cost(self, stage: str) -> float:
        return float(self._stage(stage).get("cumulative_tool_cost", 0.0))

    def next_evidence_cost(self, stage: str) -> float:
        next_stage = {"base": "screening", "screening": "verification"}.get(stage)
        if next_stage is None:
            return 0.0
        return float(self._stage(next_stage).get("tool_cost", 0.0))

    def training_row(self, stage: str) -> FactorizedGainTrainingRow:
        target = self.trajectory["hidden_outcome_for_evaluation_only"]
        return FactorizedGainTrainingRow(
            record_id=self.record_id,
            benchmark=self.benchmark,
            features=self.central_features(stage),
            success=int(target["success_label"]),
            normalized_utility=float(target["normalized_utility"]),
            normalized_action_cost=self.normalized_action_cost,
            failure_harm=self.failure_harm,
            cumulative_evidence_cost=self.cumulative_evidence_cost(stage),
        )

    def realized_gain(self, stage: str, *, cost_weight: float = 0.15) -> float:
        target = self.trajectory["hidden_outcome_for_evaluation_only"]
        return float(target["base_gain"]) - cost_weight * self.cumulative_evidence_cost(
            stage
        )

    def central_features(self, stage: str) -> dict[str, float]:
        self._validate_stage(stage)
        features = self._base_features()
        if stage == "base":
            features.update(self._prior_summary("screening"))
            features.update(self._prior_summary("verification"))
            features.update(self._mean_prior_sample("screening"))
        elif stage == "screening":
            features.update(self._observed_features("screening"))
            features.update(self._prior_summary("verification"))
            features.update(self._mean_prior_sample("verification"))
        else:
            features.update(self._observed_features("screening"))
            features.update(self._observed_features("verification"))
        return features

    def current_feature_views(self, stage: str) -> list[dict[str, float]]:
        self._validate_stage(stage)
        if stage == "base":
            return self._prior_conditioned_views("base", "screening")
        if stage == "screening":
            return self._observed_perturbation_views("screening")
        return self._observed_perturbation_views("verification")

    def counterfactual_next_stage_views(self, stage: str) -> list[dict[str, float]]:
        if stage == "base":
            return self._counterfactual_views("screening", "screening")
        if stage == "screening":
            return self._counterfactual_views("verification", "verification")
        raise ValueError("verification is terminal and has no next evidence stage")

    def _base_features(self) -> dict[str, float]:
        record = self.base_record
        excluded = set(record.get("metadata", {}).get("excluded_oracle_features", ()))
        output = {}
        for name, value in record.get("features", {}).items():
            if name in excluded or str(name).startswith("tool_"):
                continue
            if _numeric(value):
                output[f"base__{name}"] = float(value)
        return output

    def _prior_summary(self, stage: str) -> dict[str, float]:
        views = self._prior_views(stage)
        values_by_name: dict[str, list[float]] = {}
        for view in views:
            outputs = view["visible_observation"].get("visible_outputs", {})
            for name, value in outputs.items():
                if _numeric(value):
                    values_by_name.setdefault(str(name), []).append(float(value))
        output = {}
        for name, values in values_by_name.items():
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            output[f"prior_{stage}__{name}__mean"] = mean
            output[f"prior_{stage}__{name}__variance"] = variance
        return output

    def _mean_prior_sample(self, stage: str) -> dict[str, float]:
        views = self._prior_views(stage)
        values_by_name: dict[str, list[float]] = {}
        for view in views:
            for name, value in view["visible_observation"].get(
                "visible_outputs", {}
            ).items():
                if _numeric(value):
                    values_by_name.setdefault(str(name), []).append(float(value))
        output = {
            f"forecast_{stage}__{name}": sum(values) / len(values)
            for name, values in values_by_name.items()
        }
        canonical = [
            self._canonical_tool_outputs(
                view["visible_observation"].get("visible_outputs", {}),
                prior=True,
            )
            for view in views
        ]
        for name in ("margin", "positive_probability", "standard_error"):
            output[f"signal_{stage}__{name}"] = sum(row[name] for row in canonical) / len(
                canonical
            )
        return output

    def _observed_features(self, stage: str) -> dict[str, float]:
        observation = self._stage(stage)["visible_observation"]
        output = {}
        for name, value in observation.get("visible_outputs", {}).items():
            if _numeric(value):
                output[f"observed_{stage}__{name}"] = float(value)
        record_features = observation.get("record", {}).get("features", {})
        for name in (
            "tool_uncertainty",
            "tool_fidelity",
            "tool_positive_probability",
            "tool_estimated_margin",
            "tool_estimated_primary",
            "tool_estimated_secondary",
        ):
            value = record_features.get(name)
            if _numeric(value):
                output[f"observed_{stage}__{name}"] = float(value)
        canonical = self._canonical_tool_outputs(
            observation.get("visible_outputs", {}), prior=False
        )
        for name, value in canonical.items():
            output[f"signal_{stage}__{name}"] = value
        return output

    def _prior_conditioned_views(self, state_stage: str, prior_stage: str):
        central = self.central_features(state_stage)
        output = []
        for view in self._prior_views(prior_stage):
            features = dict(central)
            for name, value in view["visible_observation"].get(
                "visible_outputs", {}
            ).items():
                if _numeric(value):
                    features[f"forecast_{prior_stage}__{name}"] = float(value)
            canonical = self._canonical_tool_outputs(
                view["visible_observation"].get("visible_outputs", {}),
                prior=True,
            )
            for name, value in canonical.items():
                features[f"signal_{prior_stage}__{name}"] = value
            output.append(features)
        return output

    def _counterfactual_views(self, target_stage: str, prior_stage: str):
        central = self.central_features(target_stage)
        observed_prefix = f"observed_{target_stage}__"
        output = []
        for view in self._prior_views(prior_stage):
            features = dict(central)
            for name in tuple(features):
                if name.startswith(observed_prefix):
                    features.pop(name)
            for name, value in view["visible_observation"].get(
                "visible_outputs", {}
            ).items():
                if _numeric(value):
                    features[f"observed_{target_stage}__{name}"] = float(value)
            canonical = self._canonical_tool_outputs(
                view["visible_observation"].get("visible_outputs", {}),
                prior=True,
            )
            for name, value in canonical.items():
                features[f"signal_{target_stage}__{name}"] = value
            output.append(features)
        return output

    def _observed_perturbation_views(self, stage: str):
        central = self.central_features(stage)
        uncertainty = float(
            self._stage(stage)["visible_observation"]
            .get("record", {})
            .get("features", {})
            .get("tool_uncertainty", 0.0)
        )
        output = []
        for quantile in (-1.0, 0.0, 1.0):
            features = dict(central)
            for name in tuple(features):
                if name.startswith(f"observed_{stage}__") and any(
                    token in name
                    for token in ("margin", "primary", "secondary", "estimate")
                ):
                    features[name] = features[name] + quantile * uncertainty
            margin_name = f"signal_{stage}__margin"
            standard_error = features.get(
                f"signal_{stage}__standard_error", uncertainty
            )
            if margin_name in features:
                features[margin_name] += quantile * standard_error
            features[f"observed_{stage}__perturbation_quantile"] = quantile
            output.append(features)
        return output

    def _canonical_tool_outputs(
        self,
        outputs: Mapping[str, object],
        *,
        prior: bool,
    ) -> dict[str, float]:
        if prior:
            return {
                "margin": float(outputs["predicted_task_native_margin"]),
                "positive_probability": _unit(outputs["probability_positive_margin"]),
                "standard_error": max(1e-8, float(outputs["train_residual_scale"])),
            }
        if self.benchmark == "discover_unique":
            margin = float(outputs["estimated_composite_score"]) - float(
                outputs["training_only_advancement_threshold"]
            )
            probability = outputs["probability_above_advancement_threshold"]
            standard_error = outputs["declared_composite_standard_error"]
        elif self.benchmark == "extreme_properties":
            margin = outputs["estimated_worst_constraint_slack"]
            probability = outputs["probability_all_constraints_hit"]
            standard_error = outputs["declared_ratio_standard_error"]
        elif self.benchmark == "matbench_pairwise":
            margin = outputs["estimated_pairwise_margin"]
            probability = outputs["probability_chosen_is_better"]
            standard_error = outputs["declared_margin_standard_error"]
        else:
            raise ValueError(f"no canonical tool mapping for {self.benchmark}")
        return {
            "margin": float(margin),
            "positive_probability": _unit(probability),
            "standard_error": max(1e-8, float(standard_error)),
        }

    def _prior_views(self, stage: str) -> Sequence[Mapping[str, object]]:
        rows = [
            row
            for row in self.trajectory["pre_acquisition_prior_views"]
            if str(row["stage"]) == stage
        ]
        if len(rows) < 2:
            raise ValueError(f"record {self.record_id} lacks prior replicas for {stage}")
        return rows

    def _stage(self, stage: str) -> Mapping[str, object]:
        self._validate_stage(stage)
        for row in self.trajectory["post_acquisition_observed_stages"]:
            if str(row["stage"]) == stage:
                return row
        raise ValueError(f"record {self.record_id} lacks stage {stage}")

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown evidence stage {stage}")


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _unit(value: object) -> float:
    return min(1.0, max(0.0, float(value)))
