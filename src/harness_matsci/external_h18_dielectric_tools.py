from __future__ import annotations

import math
from dataclasses import replace

from .external_h14 import _load_snapshots
from .external_h14_tools import _fit_source_regressor, _predict_source
from .external_h18_dielectric_pairwise import TASK


class ExternalH18DielectricEvidenceEnvironment:
    dataset_name = "matbench_dielectric"
    task_name = TASK
    source_tool_name = "train_only_dielectric_source_model"

    @classmethod
    def load_materials(cls, snapshot_root):
        return _load_snapshots(snapshot_root)[cls.dataset_name]

    def __init__(self, material_by_id, regressors, train_targets):
        self.material_by_id = material_by_id
        self.regressors = regressors
        self.train_targets = tuple(sorted(train_targets))

    @classmethod
    def fit(cls, snapshot_root, train_records, *, seed):
        materials = cls.load_materials(snapshot_root)
        by_id = {material.row_id: material for material in materials}
        train_ids = {
            int(record.metadata[key])
            for record in train_records
            for key in ("chosen_row_id", "alternative_row_id")
        }
        train_materials = [by_id[row_id] for row_id in sorted(train_ids)]
        regressors = {
            source: _fit_source_regressor(
                train_materials, source=source, seed=seed + index * 97
            )
            for index, source in enumerate(("composition", "structure"))
        }
        return cls(by_id, regressors, [row.target for row in train_materials])

    def observe(self, record, *, source):
        return self.observe_many([record], source=source)[0]

    def observe_many(self, records, *, source):
        if source == "agent_prior":
            return [self._agent_prior(record) for record in records]
        if source not in ("composition", "structure", "high_fidelity"):
            raise ValueError(f"unknown source {source}")
        chosen = [
            self.material_by_id[int(record.metadata["chosen_row_id"])]
            for record in records
        ]
        alternatives = [
            self.material_by_id[int(record.metadata["alternative_row_id"])]
            for record in records
        ]
        if source == "high_fidelity":
            chosen_values = [material.target for material in chosen]
            alternative_values = [material.target for material in alternatives]
            residual = max(1e-6, 0.01 * _robust_scale(self.train_targets))
            residuals = [residual] * len(records)
            tool = "costed_database_measurement_replay"
        else:
            regressor = self.regressors[source]
            chosen_values = _predict_many(regressor, chosen)
            alternative_values = _predict_many(regressor, alternatives)
            residuals = [regressor.residual_scale] * len(records)
            tool = self.source_tool_name
        output = []
        for record, chosen_value, alternative_value, residual in zip(
            records, chosen_values, alternative_values, residuals
        ):
            if record.benchmark != self.task_name:
                raise ValueError(f"unsupported task {record.benchmark}")
            output.append(
                self._view(
                    record,
                    source=source,
                    chosen_value=chosen_value,
                    alternative_value=alternative_value,
                    residual=residual,
                    tool=tool,
                )
            )
        return output

    def _agent_prior(self, record):
        return replace(
            record,
            features={
                **record.features,
                "tool_estimated_margin": record.features["agent_prior_margin"],
                "tool_uncertainty": record.features["agent_prior_uncertainty"],
                "tool_source_confidence": record.features["agent_prior_confidence"],
                "tool_source_confidence_low": record.features["agent_prior_confidence_low"],
                "tool_source_confidence_high": record.features["agent_prior_confidence_high"],
                "tool_estimated_utility": record.features["agent_prior_estimated_utility"],
                "tool_estimated_utility_low": record.features["agent_prior_estimated_utility_low"],
                "tool_estimated_utility_high": record.features["agent_prior_estimated_utility_high"],
                "tool_cost": 0.0,
            },
            metadata={
                **record.metadata,
                "evidence_source": "agent_prior",
                "hidden_outcome_exposed": False,
                "outcome_defining_measurement_replayed": False,
            },
        )

    def _view(self, record, *, source, chosen_value, alternative_value, residual, tool):
        utility = _pair_utility(self.train_targets, chosen_value, alternative_value)
        utility_low = _pair_utility(
            self.train_targets, chosen_value - residual, alternative_value + residual
        )
        utility_high = _pair_utility(
            self.train_targets, chosen_value + residual, alternative_value - residual
        )
        margin = chosen_value - alternative_value
        uncertainty = math.sqrt(2.0) * residual
        scale = uncertainty + 1e-3
        confidence = _logistic(margin / scale)
        confidence_low = _logistic((margin - uncertainty) / scale)
        confidence_high = _logistic((margin + uncertainty) / scale)
        return replace(
            record,
            evidence=[*record.evidence, f"{source} dielectric margin={margin:.6f}"],
            features={
                **record.features,
                "tool_estimated_margin": margin,
                "tool_uncertainty": uncertainty,
                "tool_source_confidence": confidence,
                "tool_source_confidence_low": confidence_low,
                "tool_source_confidence_high": confidence_high,
                "tool_estimated_utility": utility,
                "tool_estimated_utility_low": min(utility, utility_low),
                "tool_estimated_utility_high": max(utility, utility_high),
                "tool_cost": 0.020 if source == "high_fidelity" else 0.002,
            },
            metadata={
                **record.metadata,
                "evidence_source": source,
                "evidence_provenance": {
                    "source_family": source,
                    "tool": tool,
                    "target_label_exposed": False,
                },
                "hidden_outcome_exposed": False,
                "outcome_defining_measurement_replayed": source == "high_fidelity",
            },
        )


def _pair_utility(train_targets, chosen, alternative):
    return max(0.0, _percentile(train_targets, chosen) - _percentile(train_targets, alternative))


def _predict_many(regressor, materials):
    start, end = regressor.feature_slice
    matrix = [material.vector[start:end] for material in materials]
    return [float(value) for value in regressor.model.predict(matrix)]


def _percentile(values, value):
    import bisect

    return (bisect.bisect_right(values, value) + 0.5) / (len(values) + 1.0)


def _robust_scale(values):
    return max(1e-6, _quantile(values, 0.75) - _quantile(values, 0.25))


def _quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _logistic(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(60.0, value)))
    exponential = math.exp(max(-60.0, value))
    return exponential / (1.0 + exponential)
