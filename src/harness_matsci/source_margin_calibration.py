from __future__ import annotations

from dataclasses import dataclass

from .source_fusion import SourceFusionObservation


@dataclass(frozen=True)
class SourceMarginCalibrator:
    models_by_task_and_source: dict[tuple[str, str], object]

    def predict_one(self, observation: SourceFusionObservation) -> SourceFusionObservation:
        columns = []
        for source_index, source in enumerate(observation.source_names):
            model = self.models_by_task_and_source[(observation.benchmark, source)]
            values = [row[source_index] for row in observation.predicted_gain]
            columns.append([float(value) for value in model.predict(values)])
        matrix = tuple(
            tuple(columns[source_index][replica] for source_index in range(len(columns)))
            for replica in range(len(observation.predicted_gain))
        )
        return SourceFusionObservation(
            record_id=observation.record_id,
            benchmark=observation.benchmark,
            source_names=observation.source_names,
            predicted_gain=matrix,
            gain=observation.gain,
        )

    def predict(self, observations: list[SourceFusionObservation]) -> list[SourceFusionObservation]:
        return [self.predict_one(observation) for observation in observations]

    def to_json(self) -> dict[str, object]:
        output = {}
        for (task, source), model in sorted(self.models_by_task_and_source.items()):
            output.setdefault(task, {})[source] = {
                "x_thresholds": [float(value) for value in model.X_thresholds_],
                "y_thresholds": [float(value) for value in model.y_thresholds_],
            }
        return {
            "type": "task_source_monotonic_margin_to_gain_calibrator",
            "models": output,
        }


def fit_source_margin_calibrator(
    observations: list[SourceFusionObservation],
    *,
    gain_lower_bound: float = -1.40,
    gain_upper_bound: float = 1.00,
) -> SourceMarginCalibrator:
    if not observations:
        raise ValueError("source margin calibration requires observations")
    from sklearn.isotonic import IsotonicRegression

    models = {}
    tasks = sorted({observation.benchmark for observation in observations})
    for task in tasks:
        rows = [observation for observation in observations if observation.benchmark == task]
        source_names = rows[0].source_names
        for source_index, source in enumerate(source_names):
            inputs = [
                sum(row.predicted_gain[replica][source_index] for replica in range(len(row.predicted_gain)))
                / len(row.predicted_gain)
                for row in rows
            ]
            targets = [row.gain for row in rows]
            model = IsotonicRegression(
                increasing=True,
                y_min=gain_lower_bound,
                y_max=gain_upper_bound,
                out_of_bounds="clip",
            )
            model.fit(inputs, targets)
            models[(task, source)] = model
    return SourceMarginCalibrator(models)
