from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness_matsci.action_distribution_decomposition import (
    categorical_action_grid,
    decompose_action_distribution,
)
from harness_matsci.lfp_choice_views import LfpChoiceEvidenceViewBuilder
from harness_matsci.llm_action_choice import LLMActionChoiceClient
from harness_matsci.metrics import binary_metrics
from harness_matsci.schema import ActionRecord


DEFAULT_MODELS = ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reasoning-effort", default="xhigh")
    args = parser.parse_args()
    records = _load_records(args.input)
    payload = run_crossed_grid(
        records,
        models=tuple(args.models),
        replicas=args.replicas,
        neighbors=args.neighbors,
        workers=args.workers,
        base_url=args.base_url,
        cache_dir=Path(args.cache_dir),
        reasoning_effort=args.reasoning_effort,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], sort_keys=True))


def run_crossed_grid(
    records: list[ActionRecord],
    *,
    models: tuple[str, ...],
    replicas: int,
    neighbors: int,
    workers: int,
    base_url: str,
    cache_dir: Path,
    reasoning_effort: str,
) -> dict[str, object]:
    if replicas < 2 or len(models) < 2:
        raise ValueError("crossed grid requires at least two models and two replicas")
    cache_dir.mkdir(parents=True, exist_ok=True)
    clients = {
        model: LLMActionChoiceClient(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
            cache_path=cache_dir / f"{model}.json",
            reasoning_effort=reasoning_effort,
            timeout=180.0,
            max_retries=3,
        )
        for model in models
    }
    by_batch = defaultdict(list)
    for record in records:
        by_batch[str(record.metadata["source_batch"])].append(record)
    jobs = []
    view_metadata = {}
    for held_batch, held_records in sorted(by_batch.items()):
        train = [
            record
            for batch, batch_records in by_batch.items()
            if batch != held_batch
            for record in batch_records
        ]
        builder = LfpChoiceEvidenceViewBuilder(train, neighbors=neighbors)
        for record in sorted(held_records, key=lambda item: item.record_id):
            views = builder.views(record)
            view_metadata[record.record_id] = [view.to_json() for view in views]
            for model in models:
                for replica_id in range(replicas):
                    for view in views:
                        jobs.append((record, view, model, replica_id))

    cells = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                clients[model].predict,
                view.record,
                replica_id=replica_id,
            ): (record, view, model, replica_id)
            for record, view, model, replica_id in jobs
        }
        for future in as_completed(futures):
            record, view, model, replica_id = futures[future]
            try:
                prediction = future.result()
            except Exception as error:
                failures.append(
                    {
                        "record_id": record.record_id,
                        "view_id": view.view_id,
                        "model": model,
                        "replica_id": replica_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            cells.append(
                {
                    "record_id": record.record_id,
                    "source_batch": record.metadata["source_batch"],
                    "label_a_better": record.label,
                    "view_id": view.view_id,
                    "agent_replica_id": f"{model}::replica-{replica_id}",
                    "model": model,
                    "replica_id": replica_id,
                    **prediction.to_json(),
                }
            )
    if failures:
        raise RuntimeError(f"crossed grid has {len(failures)} failed cells: {failures[:3]}")

    rows = _summarize_records(records, cells, models=models, replicas=replicas)
    return {
        "development_only": True,
        "protocol": "leave-one-source-batch-out evidence retrieval",
        "models": list(models),
        "replicas_per_model": replicas,
        "external_views": [
            "evidence::base",
            "tool::formulation_descriptors",
            "tool::held_batch_analogs",
        ],
        "record_count": len(records),
        "expected_cells": len(records) * len(models) * replicas * 3,
        "observed_cells": len(cells),
        "view_metadata": view_metadata,
        "cells": sorted(
            cells,
            key=lambda row: (
                row["record_id"],
                row["agent_replica_id"],
                row["view_id"],
            ),
        ),
        "records": rows,
        "summary": _summary(rows),
    }


def _summarize_records(records, cells, *, models, replicas):
    by_record = defaultdict(list)
    for cell in cells:
        by_record[cell["record_id"]].append(cell)
    agent_ids = tuple(
        f"{model}::replica-{replica_id}"
        for model in models
        for replica_id in range(replicas)
    )
    view_ids = (
        "evidence::base",
        "tool::formulation_descriptors",
        "tool::held_batch_analogs",
    )
    output = []
    for record in sorted(records, key=lambda item: item.record_id):
        lookup = {
            (cell["agent_replica_id"], cell["view_id"]): cell
            for cell in by_record[record.record_id]
        }
        expected = {(agent, view) for agent in agent_ids for view in view_ids}
        if set(lookup) != expected:
            raise ValueError(f"unbalanced grid for {record.record_id}")
        probability_grid = tuple(
            tuple(
                (
                    float(lookup[(agent, view)]["p_a_better"]),
                    1.0 - float(lookup[(agent, view)]["p_a_better"]),
                )
                for view in view_ids
            )
            for agent in agent_ids
        )
        choice_grid = tuple(
            tuple(lookup[(agent, view)]["choice"] for view in view_ids)
            for agent in agent_ids
        )
        probability_variance = decompose_action_distribution(
            probability_grid,
            action_names=("A", "B"),
        )
        choice_variance = decompose_action_distribution(
            categorical_action_grid(choice_grid, action_names=("A", "B")),
            action_names=("A", "B"),
        )
        view_probabilities = {
            view: _mean(
                float(lookup[(agent, view)]["p_a_better"])
                for agent in agent_ids
            )
            for view in view_ids
        }
        output.append(
            {
                "record_id": record.record_id,
                "source_batch": record.metadata["source_batch"],
                "label": record.label,
                "view_probabilities": view_probabilities,
                "all_view_probability": _mean(view_probabilities.values()),
                "probability_variance": probability_variance.to_json(),
                "choice_variance": choice_variance.to_json(),
            }
        )
    return output


def _summary(rows):
    labels = [row["label"] for row in rows]
    views = tuple(next(iter(rows))["view_probabilities"])
    metrics = {
        view: binary_metrics(
            labels,
            [row["view_probabilities"][view] for row in rows],
            threshold=0.5,
        )
        for view in views
    }
    metrics["all_views_mean"] = binary_metrics(
        labels,
        [row["all_view_probability"] for row in rows],
        threshold=0.5,
    )
    base_errors = [
        float((row["view_probabilities"]["evidence::base"] >= 0.5) != bool(row["label"]))
        for row in rows
    ]
    base_squared_errors = [
        (row["view_probabilities"]["evidence::base"] - row["label"]) ** 2
        for row in rows
    ]
    correlations = {}
    for representation in ("probability_variance", "choice_variance"):
        for component in (
            "internal_variance",
            "external_variance",
            "interaction_variance",
            "total_variance",
            "consensus_uncertainty",
        ):
            values = [row[representation][component] for row in rows]
            correlations[f"{representation}.{component}.binary_error"] = _correlation(
                values, base_errors
            )
            correlations[f"{representation}.{component}.squared_error"] = _correlation(
                values, base_squared_errors
            )
    return {"metrics": metrics, "diagnostic_correlations": correlations}


def _load_records(path):
    return [
        ActionRecord.from_json(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values):
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _correlation(left, right):
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


if __name__ == "__main__":
    main()
