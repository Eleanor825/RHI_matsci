from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from harness_matsci.lfp_formulation_database import (
    build_lfp_candidate_actions,
    build_lfp_pairwise_actions,
    load_lfp_formulation_experiments,
)
from harness_matsci.matbot_lfp_historical import build_matbot_lfp_historical_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    experiments = load_lfp_formulation_experiments(source)
    candidates = build_lfp_candidate_actions(experiments)
    pairs = build_lfp_pairwise_actions(candidates)
    candidate_trajectories = build_matbot_lfp_historical_trajectories(candidates)
    pair_trajectories = build_matbot_lfp_historical_trajectories(pairs)
    _write_jsonl(destination / "candidate_actions.jsonl", [row.to_json() for row in candidates])
    _write_jsonl(destination / "pairwise_actions.jsonl", [row.to_json() for row in pairs])
    _write_jsonl(
        destination / "candidate_trajectories.jsonl",
        [row.to_json() for row in candidate_trajectories],
    )
    _write_jsonl(
        destination / "pairwise_trajectories.jsonl",
        [row.to_json() for row in pair_trajectories],
    )
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "experiment_count": len(experiments),
        "experiment_families": dict(Counter(row.family for row in experiments)),
        "candidate_action_count": len(candidates),
        "candidate_tasks": dict(Counter(row.benchmark for row in candidates)),
        "pairwise_action_count": len(pairs),
        "pairwise_tasks": dict(Counter(row.benchmark for row in pairs)),
        "candidate_complete_trajectories": sum(row.complete for row in candidate_trajectories),
        "pairwise_complete_trajectories": sum(row.complete for row in pair_trajectories),
        "hidden_outcome_contract": True,
        "orientation_uses_hidden_outcomes": False,
        "confirmation_status": "real_source_development_external_validation",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
