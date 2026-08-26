from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .hurdle_gain_calibration import (
    HurdleGainObservation,
    HurdleGainPrediction,
    HurdleSourceGainCalibrator,
    fit_hurdle_gain_calibrator,
)
from .schema import ActionRecord
from .scientific_gain import EstimatedGainProjector
from .source_fusion import (
    SourceFusionObservation,
    SourceReliabilityFuser,
    fit_source_reliability_fuser,
)
from .source_margin_calibration import (
    SourceMarginCalibrator,
    fit_source_margin_calibrator,
)


@dataclass(frozen=True)
class SourceStateModel:
    source_names: tuple[str, ...]
    margin_calibrator: SourceMarginCalibrator | None
    source_fuser: SourceReliabilityFuser
    hurdle_calibrator: HurdleSourceGainCalibrator

    def predict(
        self, observations: list[SourceFusionObservation]
    ) -> list[HurdleGainPrediction]:
        calibrated = (
            self.margin_calibrator.predict(observations)
            if self.margin_calibrator is not None
            else observations
        )
        fused = self.source_fuser.predict(calibrated)
        return self.hurdle_calibrator.predict(
            [
                HurdleGainObservation(
                    record_id=row.record_id,
                    benchmark=row.benchmark,
                    mean_gain=row.mean_gain,
                    internal_variance=row.internal_variance,
                    external_variance=row.external_variance,
                    gain=observation.gain,
                    interaction_variance=row.interaction_variance,
                    internal_boundary_instability=row.internal_boundary_instability,
                    external_boundary_instability=row.external_boundary_instability,
                )
                for row, observation in zip(fused, observations)
            ]
        )

    def to_json(self) -> dict[str, object]:
        return {
            "source_names": list(self.source_names),
            "margin_calibrator": (
                self.margin_calibrator.to_json()
                if self.margin_calibrator is not None
                else None
            ),
            "source_fuser": self.source_fuser.to_json(),
            "hurdle_calibrator": self.hurdle_calibrator.to_json(),
        }


def build_source_observations(
    environment,
    records: list[ActionRecord],
    target_by_id,
    *,
    source_names: tuple[str, ...],
    gain_projector: EstimatedGainProjector | None = None,
) -> list[SourceFusionObservation]:
    views_by_source = {}
    for source in source_names:
        if hasattr(environment, "observe_many"):
            views_by_source[source] = environment.observe_many(records, source=source)
        else:
            views_by_source[source] = [
                environment.observe(record, source=source) for record in records
            ]
    observations = []
    for record_index, record in enumerate(records):
        if gain_projector is None:
            matrix = [
                [
                    float(
                        views_by_source[source][record_index].features[
                            "tool_estimated_margin"
                        ]
                    )
                    + replica_offset
                    * float(
                        views_by_source[source][record_index].features["tool_uncertainty"]
                    )
                    for source in source_names
                ]
                for replica_offset in (-1.0, 0.0, 1.0)
            ]
        else:
            utility_features = (
                "tool_estimated_utility_low",
                "tool_estimated_utility",
                "tool_estimated_utility_high",
            )
            confidence_features = (
                "tool_source_confidence_low",
                "tool_source_confidence",
                "tool_source_confidence_high",
            )
            matrix = [
                [
                    gain_projector.project(
                        record,
                        estimated_utility=float(
                            views_by_source[source][record_index].features[utility_feature]
                        ),
                        success_probability=float(
                            views_by_source[source][record_index].features[confidence_feature]
                        ),
                    )
                    for source in source_names
                ]
                for utility_feature, confidence_feature in zip(
                    utility_features, confidence_features
                )
            ]
        observations.append(
            SourceFusionObservation(
                record_id=record.record_id,
                benchmark=record.benchmark,
                source_names=source_names,
                predicted_gain=tuple(tuple(row) for row in matrix),
                gain=target_by_id[record.record_id].gain,
            )
        )
    return observations


def fit_source_state_model(
    observations: list[SourceFusionObservation],
    *,
    use_source_variance: bool = True,
    beta: float = 0.10,
    tail_weight: float = 0.50,
    calibrate_source_margins: bool = True,
    use_decision_instability: bool = False,
) -> SourceStateModel:
    if not observations:
        raise ValueError("source-state fitting requires observations")
    source_names = observations[0].source_names
    if any(row.source_names != source_names for row in observations):
        raise ValueError("source-state observations must share source ordering")
    margin_calibrator = (
        fit_source_margin_calibrator(observations)
        if calibrate_source_margins
        else None
    )
    calibrated = (
        margin_calibrator.predict(observations)
        if margin_calibrator is not None
        else observations
    )
    source_fuser = fit_source_reliability_fuser(calibrated, fit_affine=False)
    fused = source_fuser.predict(calibrated)
    hurdle_calibrator = fit_hurdle_gain_calibrator(
        [
            HurdleGainObservation(
                record_id=row.record_id,
                benchmark=row.benchmark,
                mean_gain=row.mean_gain,
                internal_variance=row.internal_variance,
                external_variance=row.external_variance,
                gain=observation.gain,
                interaction_variance=row.interaction_variance,
                internal_boundary_instability=row.internal_boundary_instability,
                external_boundary_instability=row.external_boundary_instability,
            )
            for row, observation in zip(fused, observations)
        ],
        use_source_variance=use_source_variance,
        use_decision_instability=use_decision_instability,
        beta=beta,
        tail_weight=tail_weight,
    )
    return SourceStateModel(
        source_names=source_names,
        margin_calibrator=margin_calibrator,
        source_fuser=source_fuser,
        hurdle_calibrator=hurdle_calibrator,
    )


def cross_fitted_source_state_predictions(
    observations: list[SourceFusionObservation],
    *,
    folds: int = 5,
    seed: int = 1401,
    use_source_variance: bool = True,
    beta: float = 0.10,
    tail_weight: float = 0.50,
    calibrate_source_margins: bool = True,
    use_decision_instability: bool = False,
) -> list[HurdleGainPrediction]:
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    assignments = [
        _fold_assignment(row.record_id, row.benchmark, folds=folds, seed=seed)
        for row in observations
    ]
    predictions: list[HurdleGainPrediction | None] = [None] * len(observations)
    for fold in range(folds):
        train = [row for row, assignment in zip(observations, assignments) if assignment != fold]
        held_indices = [
            index for index, assignment in enumerate(assignments) if assignment == fold
        ]
        if not held_indices:
            continue
        model = fit_source_state_model(
            train,
            use_source_variance=use_source_variance,
            beta=beta,
            tail_weight=tail_weight,
            calibrate_source_margins=calibrate_source_margins,
            use_decision_instability=use_decision_instability,
        )
        held_predictions = model.predict([observations[index] for index in held_indices])
        for index, prediction in zip(held_indices, held_predictions):
            predictions[index] = prediction
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("cross-fitting did not predict every observation")
    return [prediction for prediction in predictions if prediction is not None]


def _fold_assignment(record_id: str, benchmark: str, *, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{benchmark}:{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds
