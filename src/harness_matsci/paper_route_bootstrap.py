from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import expected_calibration_error, threshold_for_selective_risk
from .features import Standardizer, clamp, matrix_from_records
from .io import write_json
from .metrics import binary_metrics
from .models import LogisticGate
from .paper_bootstrap import (
    DEFAULT_PAPER_ACTIONS_PATH,
    PAPER_V2_FEATURES,
    evidence_heuristic_baseline,
    load_paper_action_records,
    split_records_by_group,
    summarize_paper_records,
    train_gate_with_features,
    verbal_confidence_baseline,
)
from .routing import route_record
from .schema import ActionRecord


ROUTE_LABELS = ("retrieve_more", "proceed", "experiment", "abstain")


def paper_route_label(record: ActionRecord) -> str:
    raw_action_type = str(record.metadata.get("raw_action_type") or record.action_type)
    if raw_action_type in {"ask_more", "retrieve_more", "summarize_literature"}:
        return "retrieve_more"
    if raw_action_type in {"recommend_experiment", "execute_tool"}:
        return "experiment"
    if raw_action_type in {"commit_decision", "choose_candidate"}:
        return "proceed"
    if raw_action_type == "abstain":
        return "abstain"
    return "retrieve_more"


@dataclass
class RouteEnsemble:
    feature_names: list[str]
    standardizer: Standardizer
    models: dict[str, LogisticGate]
    threshold: float
    calibration: dict[str, float]

    def predict_route_probabilities(self, records: list[ActionRecord]) -> list[dict[str, float]]:
        matrix = matrix_from_records(records, self.feature_names)
        rows = self.standardizer.transform(matrix.x)
        route_probabilities: list[dict[str, float]] = []
        for row in rows:
            route_probabilities.append({route: model.predict_proba_row(row) for route, model in self.models.items()})
        return route_probabilities

    def predict_routes(self, records: list[ActionRecord]) -> tuple[list[str], list[float], list[dict[str, float]]]:
        route_probabilities = self.predict_route_probabilities(records)
        predicted_routes: list[str] = []
        confidences: list[float] = []
        for probabilities in route_probabilities:
            route, confidence = max(probabilities.items(), key=lambda item: item[1])
            if confidence < self.threshold:
                route = "abstain"
            predicted_routes.append(route)
            confidences.append(float(confidence))
        return predicted_routes, confidences, route_probabilities

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "standardizer": self.standardizer.to_json(),
            "models": {route: model.to_json() for route, model in self.models.items()},
            "threshold": self.threshold,
            "calibration": dict(self.calibration),
        }


def train_route_ensemble(
    train_records: list[ActionRecord],
    val_records: list[ActionRecord],
    *,
    feature_names: list[str] | None = None,
    alpha: float = 0.1,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> RouteEnsemble:
    if not train_records:
        raise ValueError("train_records cannot be empty")
    if not val_records:
        raise ValueError("val_records cannot be empty")
    names = feature_names or list(PAPER_V2_FEATURES)
    train_matrix = matrix_from_records(train_records, names)
    standardizer = Standardizer().fit(train_matrix.x)
    x_train = standardizer.transform(train_matrix.x)

    models: dict[str, LogisticGate] = {}
    for route in ROUTE_LABELS:
        labels = [1 if paper_route_label(record) == route else 0 for record in train_records]
        positives = sum(labels)
        negatives = len(labels) - positives
        if positives == 0:
            # Keep a degenerate model rather than crashing; it will predict near-zero.
            sample_weight = [1.0 for _ in labels]
        else:
            positive_weight = negatives / max(1, positives)
            sample_weight = [positive_weight if label else 1.0 for label in labels]
        model = LogisticGate.fresh(names)
        model.fit(
            x_train,
            labels,
            sample_weight=sample_weight,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        models[route] = model

    ensemble = RouteEnsemble(feature_names=names, standardizer=standardizer, models=models, threshold=0.5, calibration={})
    val_predictions, val_confidences, _ = ensemble.predict_routes(val_records)
    val_labels = [paper_route_label(record) for record in val_records]
    correctness = [1 if pred == label else 0 for pred, label in zip(val_predictions, val_labels)]
    calibration = threshold_for_selective_risk(correctness, val_confidences, alpha=alpha)
    ensemble.threshold = calibration["threshold"]
    ensemble.calibration = calibration
    return ensemble


def evaluate_route_policy(
    records: list[ActionRecord],
    predicted_routes: list[str],
    confidences: list[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = [paper_route_label(record) for record in records]
    if len(labels) != len(predicted_routes) or len(labels) != len(confidences):
        raise ValueError("records, predicted_routes, and confidences must have same length")
    correctness = [1 if pred == label else 0 for pred, label in zip(predicted_routes, labels)]
    metrics = binary_metrics(correctness, confidences, threshold)
    route_accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    route_counts = Counter(predicted_routes)
    label_counts = Counter(labels)
    selected = [idx for idx, conf in enumerate(confidences) if conf >= threshold]
    selected_route_accuracy = sum(correctness[idx] for idx in selected) / len(selected) if selected else 0.0
    selected_route_risk = 1.0 - selected_route_accuracy if selected else 0.0
    return {
        "route_accuracy": route_accuracy,
        "selected_route_accuracy": selected_route_accuracy,
        "selected_route_risk": selected_route_risk,
        "coverage": metrics["coverage"],
        "selective_accuracy": metrics["selective_accuracy"],
        "selective_risk": metrics["selective_risk"],
        "confidently_wrong_rate": metrics["confidently_wrong_rate"],
        "ece": expected_calibration_error(correctness, confidences),
        "brier": metrics["brier"],
        "log_loss": metrics["log_loss"],
        "route_counts": dict(sorted(route_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
    }


def run_paper_route_bootstrap_experiment(
    data_path: str | Path = DEFAULT_PAPER_ACTIONS_PATH,
    *,
    workdir: str | Path | None = None,
    seed: int = 7,
    alpha: float = 0.1,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    budget_fraction: float = 0.1,
) -> dict[str, Any]:
    records = load_paper_action_records(data_path)
    train_records, val_records, test_records, split_summary = split_records_by_group(
        records,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    gate = train_gate_with_features(
        train_records,
        val_records,
        feature_names=list(PAPER_V2_FEATURES),
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    gate_val = gate.decisions(val_records)
    gate_test = gate.decisions(test_records)
    gate_val_routes = [decision.route for decision in gate_val]
    gate_test_routes = [decision.route for decision in gate_test]
    gate_val_confidences = [decision.reliability for decision in gate_val]
    gate_test_confidences = [decision.reliability for decision in gate_test]
    gate_val_route_metrics = evaluate_route_policy(
        val_records,
        gate_val_routes,
        gate_val_confidences,
        threshold=gate.threshold,
    )
    gate_test_route_metrics = evaluate_route_policy(
        test_records,
        gate_test_routes,
        gate_test_confidences,
        threshold=gate.threshold,
    )

    route_ensemble = train_route_ensemble(
        train_records,
        val_records,
        feature_names=list(PAPER_V2_FEATURES),
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    route_val_pred, route_val_conf, _ = route_ensemble.predict_routes(val_records)
    route_test_pred, route_test_conf, route_test_probs = route_ensemble.predict_routes(test_records)
    route_val_metrics = evaluate_route_policy(
        val_records,
        route_val_pred,
        route_val_conf,
        threshold=route_ensemble.threshold,
    )
    route_test_metrics = evaluate_route_policy(
        test_records,
        route_test_pred,
        route_test_conf,
        threshold=route_ensemble.threshold,
    )

    versions = {
        "v2_heuristic_router": {
            "gate": gate.to_json(),
            "gate_val": {
                "metrics": gate_val_route_metrics,
                "decision_metrics": gate_val_route_metrics,
                "calibration": gate.calibration,
            },
            "gate_test": {
                "metrics": gate_test_route_metrics,
                "decision_metrics": gate_test_route_metrics,
                "calibration": gate.calibration,
            },
            "route_policy": "heuristic route_record",
        },
        "v3_route_aware_paper_harness": {
            "gate": gate.to_json(),
            "gate_val": {
                "metrics": gate_val_route_metrics,
                "decision_metrics": gate_val_route_metrics,
                "calibration": gate.calibration,
            },
            "gate_test": {
                "metrics": gate_test_route_metrics,
                "decision_metrics": gate_test_route_metrics,
                "calibration": gate.calibration,
            },
            "route_policy": {
                "feature_names": list(route_ensemble.feature_names),
                "threshold": route_ensemble.threshold,
                "calibration": route_ensemble.calibration,
                "models": {route: model.to_json() for route, model in route_ensemble.models.items()},
            },
            "route_val": route_val_metrics,
            "route_test": route_test_metrics,
        },
    }

    selected_version = _select_route_version(versions)
    report = {
        "experiment": {
            "name": "paper_route_bootstrap_v2",
            "objective": "train a route-aware runtime uncertainty harness from historical electrolyte papers",
            "data_path": str(data_path),
            "seed": seed,
            "alpha": alpha,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "budget_fraction": budget_fraction,
        },
        "dataset": summarize_paper_records(records),
        "split": split_summary,
        "versions": versions,
        "selected_version": selected_version,
        "baselines": {
            "verbal_confidence": verbal_confidence_baseline(test_records, alpha=alpha, budget_fraction=budget_fraction),
            "evidence_heuristic": evidence_heuristic_baseline(test_records, alpha=alpha, budget_fraction=budget_fraction),
        },
        "notes": [
            "This second version adds route supervision on top of the reliability gate.",
            "Route labels are derived from historical paper action types and are therefore weak supervision.",
            "The route-aware head is meant to approximate the full proposal's proceed/retrieve/experiment/abstain routing layer.",
        ],
    }

    if workdir is not None:
        target = Path(workdir)
        target.mkdir(parents=True, exist_ok=True)
        write_json(report, target / "summary.json")
        write_json(route_ensemble.to_json(), target / "route_ensemble.json")
        write_json(gate.to_json(), target / "gate.json")
    return report


def _select_route_version(versions: dict[str, Any]) -> str:
    def score(item: tuple[str, Any]) -> tuple[float, float, float]:
        version_name, payload = item
        gate_val = payload["gate_val"]["metrics"]
        route_val = payload.get("route_val", gate_val)
        # Lower is better: gate risk + route selective risk, with a small coverage tie-break.
        combined = float(gate_val["selective_risk"]) + float(route_val["selective_risk"])
        coverage_penalty = -float(route_val["coverage"])
        return combined, coverage_penalty, -float(route_val.get("route_accuracy", gate_val.get("route_accuracy", 0.0)))

    return min(versions.items(), key=score)[0]

