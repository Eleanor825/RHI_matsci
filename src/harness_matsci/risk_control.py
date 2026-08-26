from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskControlledSelection:
    threshold: float
    selected_count: int
    selected_fraction: float
    empirical_risk: float
    risk_upper_bound: float
    alpha: float
    delta: float
    candidate_prefixes: int

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)


def select_risk_controlled_threshold(
    probabilities: list[float],
    worthy_labels: list[int],
    *,
    alpha: float = 0.1,
    delta: float = 0.05,
) -> RiskControlledSelection:
    if len(probabilities) != len(worthy_labels):
        raise ValueError("probabilities and labels must align")
    if not probabilities:
        raise ValueError("risk control requires non-empty calibration data")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must lie in (0, 1)")
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    per_prefix_delta = delta / len(order)
    errors = 0
    best: RiskControlledSelection | None = None
    for prefix, index in enumerate(order, 1):
        errors += 1 - int(worthy_labels[index])
        next_is_error = prefix < len(order) and int(worthy_labels[order[prefix]]) == 0
        if prefix < len(order) and not next_is_error:
            continue
        upper = binomial_upper_confidence(errors, prefix, per_prefix_delta)
        if upper <= alpha:
            next_probability = probabilities[order[prefix]] if prefix < len(order) else -1.0
            threshold = (probabilities[index] + next_probability) / 2.0
            best = RiskControlledSelection(
                threshold=threshold,
                selected_count=prefix,
                selected_fraction=prefix / len(order),
                empirical_risk=errors / prefix,
                risk_upper_bound=upper,
                alpha=alpha,
                delta=delta,
                candidate_prefixes=len(order),
            )
    if best is not None:
        return best
    return RiskControlledSelection(
        threshold=1.0 + 1e-12,
        selected_count=0,
        selected_fraction=0.0,
        empirical_risk=0.0,
        risk_upper_bound=0.0,
        alpha=alpha,
        delta=delta,
        candidate_prefixes=len(order),
    )


def select_risk_controlled_grid(
    probabilities: list[float],
    worthy_labels: list[int],
    *,
    coverages: tuple[float, ...],
    alpha: float = 0.1,
    delta: float = 0.05,
) -> RiskControlledSelection:
    audit = audit_risk_controlled_grid(
        probabilities,
        worthy_labels,
        coverages=coverages,
        alpha=alpha,
        delta=delta,
    )
    best: RiskControlledSelection | None = None
    for candidate in audit:
        if candidate["certified"]:
            best = RiskControlledSelection(
                threshold=float(candidate["threshold"]),
                selected_count=int(candidate["selected_count"]),
                selected_fraction=float(candidate["selected_fraction"]),
                empirical_risk=float(candidate["empirical_risk"]),
                risk_upper_bound=float(candidate["risk_upper_bound"]),
                alpha=alpha,
                delta=delta,
                candidate_prefixes=len(audit),
            )
    if best is not None:
        return best
    return RiskControlledSelection(
        threshold=1.0 + 1e-12,
        selected_count=0,
        selected_fraction=0.0,
        empirical_risk=0.0,
        risk_upper_bound=0.0,
        alpha=alpha,
        delta=delta,
        candidate_prefixes=len(audit),
    )


def audit_risk_controlled_grid(
    probabilities: list[float],
    worthy_labels: list[int],
    *,
    coverages: tuple[float, ...],
    alpha: float = 0.1,
    delta: float = 0.05,
) -> list[dict[str, float | int | bool]]:
    if len(probabilities) != len(worthy_labels):
        raise ValueError("probabilities and labels must align")
    if not probabilities:
        raise ValueError("risk control requires non-empty calibration data")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must lie in (0, 1)")
    if not coverages or any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("coverage candidates must lie in (0, 1]")
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    candidate_counts = sorted(
        {max(1, min(len(order), math.ceil(len(order) * coverage))) for coverage in coverages}
    )
    per_candidate_delta = delta / len(candidate_counts)
    output: list[dict[str, float | int | bool]] = []
    for count in candidate_counts:
        selected = order[:count]
        errors = sum(1 - int(worthy_labels[index]) for index in selected)
        upper = binomial_upper_confidence(errors, count, per_candidate_delta)
        next_probability = probabilities[order[count]] if count < len(order) else -1.0
        threshold = (probabilities[order[count - 1]] + next_probability) / 2.0
        output.append(
            {
                "selected_count": count,
                "selected_fraction": count / len(order),
                "errors": errors,
                "empirical_risk": errors / count,
                "risk_upper_bound": upper,
                "threshold": threshold,
                "per_candidate_delta": per_candidate_delta,
                "certified": upper <= alpha,
            }
        )
    return output


def select_bounded_risk_grid(
    scores: list[float],
    losses: list[float],
    *,
    coverages: tuple[float, ...],
    alpha: float = 0.1,
    delta: float = 0.05,
) -> RiskControlledSelection:
    if len(scores) != len(losses):
        raise ValueError("scores and losses must align")
    if not scores:
        raise ValueError("bounded risk control requires non-empty calibration data")
    if any(not 0.0 <= loss <= 1.0 for loss in losses):
        raise ValueError("bounded losses must lie in [0, 1]")
    if not coverages or any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("coverage candidates must lie in (0, 1]")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    candidate_counts = sorted(
        {max(1, min(len(order), math.ceil(len(order) * coverage))) for coverage in coverages}
    )
    per_candidate_delta = delta / len(candidate_counts)
    best: RiskControlledSelection | None = None
    for count in candidate_counts:
        selected_losses = [losses[index] for index in order[:count]]
        empirical = sum(selected_losses) / count
        upper = empirical_bernstein_upper_confidence(selected_losses, per_candidate_delta)
        if upper <= alpha:
            next_score = scores[order[count]] if count < len(order) else -1.0
            threshold = (scores[order[count - 1]] + next_score) / 2.0
            best = RiskControlledSelection(
                threshold=threshold,
                selected_count=count,
                selected_fraction=count / len(order),
                empirical_risk=empirical,
                risk_upper_bound=upper,
                alpha=alpha,
                delta=delta,
                candidate_prefixes=len(candidate_counts),
            )
    if best is not None:
        return best
    return RiskControlledSelection(
        threshold=1.0 + 1e-12,
        selected_count=0,
        selected_fraction=0.0,
        empirical_risk=0.0,
        risk_upper_bound=0.0,
        alpha=alpha,
        delta=delta,
        candidate_prefixes=len(candidate_counts),
    )


def select_hoeffding_risk_grid(
    scores: list[float],
    losses: list[float],
    *,
    coverages: tuple[float, ...],
    alpha: float = 0.1,
    delta: float = 0.05,
) -> RiskControlledSelection:
    if len(scores) != len(losses):
        raise ValueError("scores and losses must align")
    if not scores:
        raise ValueError("bounded risk control requires non-empty calibration data")
    if any(not 0.0 <= loss <= 1.0 for loss in losses):
        raise ValueError("bounded losses must lie in [0, 1]")
    if not coverages or any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("coverage candidates must lie in (0, 1]")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    candidate_counts = sorted(
        {max(1, min(len(order), math.ceil(len(order) * coverage))) for coverage in coverages}
    )
    per_candidate_delta = delta / len(candidate_counts)
    best: RiskControlledSelection | None = None
    for count in candidate_counts:
        selected_losses = [losses[index] for index in order[:count]]
        empirical = sum(selected_losses) / count
        upper = hoeffding_upper_confidence(selected_losses, per_candidate_delta)
        if upper <= alpha:
            next_score = scores[order[count]] if count < len(order) else -1.0
            threshold = (scores[order[count - 1]] + next_score) / 2.0
            best = RiskControlledSelection(
                threshold=threshold,
                selected_count=count,
                selected_fraction=count / len(order),
                empirical_risk=empirical,
                risk_upper_bound=upper,
                alpha=alpha,
                delta=delta,
                candidate_prefixes=len(candidate_counts),
            )
    if best is not None:
        return best
    return RiskControlledSelection(
        threshold=1.0 + 1e-12,
        selected_count=0,
        selected_fraction=0.0,
        empirical_risk=0.0,
        risk_upper_bound=0.0,
        alpha=alpha,
        delta=delta,
        candidate_prefixes=len(candidate_counts),
    )


def hoeffding_upper_confidence(losses: list[float], delta: float) -> float:
    if not losses:
        return 0.0
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if any(not 0.0 <= loss <= 1.0 for loss in losses):
        raise ValueError("bounded losses must lie in [0, 1]")
    empirical = sum(losses) / len(losses)
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * len(losses)))
    return min(1.0, empirical + radius)


def empirical_bernstein_upper_confidence(losses: list[float], delta: float) -> float:
    if not losses:
        return 0.0
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if any(not 0.0 <= loss <= 1.0 for loss in losses):
        raise ValueError("bounded losses must lie in [0, 1]")
    count = len(losses)
    empirical = sum(losses) / count
    if count == 1:
        return min(1.0, empirical + math.sqrt(math.log(1.0 / delta) / 2.0))
    variance = sum((loss - empirical) ** 2 for loss in losses) / (count - 1)
    log_term = math.log(3.0 / delta)
    radius = math.sqrt(2.0 * variance * log_term / count)
    radius += 3.0 * log_term / (count - 1)
    return min(1.0, empirical + radius)


def empirical_bernstein_lower_confidence(
    values: list[float],
    delta: float,
    *,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """One-sided empirical-Bernstein lower bound for bounded observations."""
    if not values:
        return lower_bound
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not lower_bound < upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    tolerance = 1e-12
    if any(value < lower_bound - tolerance or value > upper_bound + tolerance for value in values):
        raise ValueError("values must lie within the declared bounds")
    scale = upper_bound - lower_bound
    normalized = [
        min(1.0, max(0.0, (value - lower_bound) / scale)) for value in values
    ]
    count = len(normalized)
    empirical = sum(normalized) / count
    if count == 1:
        radius = math.sqrt(math.log(1.0 / delta) / 2.0)
    else:
        variance = sum((value - empirical) ** 2 for value in normalized) / (count - 1)
        log_term = math.log(3.0 / delta)
        radius = math.sqrt(2.0 * variance * log_term / count)
        radius += 3.0 * log_term / (count - 1)
    return max(lower_bound, lower_bound + scale * (empirical - radius))


def binomial_upper_confidence(errors: int, trials: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound solved without SciPy."""
    if trials <= 0:
        return 0.0
    if errors < 0 or errors > trials:
        raise ValueError("errors must be between zero and trials")
    if errors == trials:
        return 1.0
    low = errors / trials
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        cdf = _binomial_cdf(errors, trials, midpoint)
        if cdf > delta:
            low = midpoint
        else:
            high = midpoint
    return high


def selected_indices(probabilities: list[float], threshold: float) -> list[int]:
    return [index for index, probability in enumerate(probabilities) if probability >= threshold]


def _binomial_cdf(errors: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if errors >= trials else 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [trials * log_q]
    current = terms[0]
    for count in range(1, errors + 1):
        current += (
            math.log(trials - count + 1)
            - math.log(count)
            + log_p
            - log_q
        )
        terms.append(current)
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)
