from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceVariance:
    internal_variance: float
    external_variance: float
    interaction_variance: float
    total_variance: float
    agent_replicas: int
    evidence_views: int

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)


def decompose_source_variance(predicted_gain: list[list[float]]) -> SourceVariance:
    """Orthogonally decompose a balanced agent-replica x evidence-view matrix.

    Internal variance is the replica main effect, external variance is the
    evidence-view main effect, and interaction variance is the non-additive
    replica-by-view residual. The three components sum exactly to the total
    population variance of the balanced matrix.
    """
    if len(predicted_gain) < 2:
        raise ValueError("at least two agent replicas are required")
    widths = {len(row) for row in predicted_gain}
    if len(widths) != 1 or not widths or next(iter(widths)) < 2:
        raise ValueError("a balanced matrix with at least two evidence views is required")
    evidence_views = next(iter(widths))
    replica_means = [sum(row) / evidence_views for row in predicted_gain]
    column_means = [
        sum(row[column] for row in predicted_gain) / len(predicted_gain)
        for column in range(evidence_views)
    ]
    grand_mean = sum(replica_means) / len(replica_means)
    internal = _population_variance(replica_means)
    external = _population_variance(column_means)
    interaction = sum(
        (
            predicted_gain[replica][column]
            - replica_means[replica]
            - column_means[column]
            + grand_mean
        )
        ** 2
        for replica in range(len(predicted_gain))
        for column in range(evidence_views)
    ) / (len(predicted_gain) * evidence_views)
    return SourceVariance(
        internal_variance=internal,
        external_variance=external,
        interaction_variance=interaction,
        total_variance=internal + external + interaction,
        agent_replicas=len(predicted_gain),
        evidence_views=evidence_views,
    )


def _population_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
