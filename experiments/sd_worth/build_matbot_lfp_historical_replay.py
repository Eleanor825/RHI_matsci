from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from harness_matsci.matbot_lfp_historical import (
    build_matbot_lfp_historical_choices,
    build_matbot_lfp_historical_trajectories,
    load_matbot_lfp_historical_actions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--choice-trajectories", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    records = load_matbot_lfp_historical_actions(source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )
    trajectories = build_matbot_lfp_historical_trajectories(records)
    Path(args.trajectories).write_text(
        "\n".join(
            json.dumps(trajectory.to_json(), ensure_ascii=False, sort_keys=True)
            for trajectory in trajectories
        )
        + "\n",
        encoding="utf-8",
    )
    choices = build_matbot_lfp_historical_choices(records)
    Path(args.choices).write_text(
        "\n".join(
            json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True)
            for record in choices
        )
        + "\n",
        encoding="utf-8",
    )
    choice_trajectories = build_matbot_lfp_historical_trajectories(choices)
    Path(args.choice_trajectories).write_text(
        "\n".join(
            json.dumps(trajectory.to_json(), ensure_ascii=False, sort_keys=True)
            for trajectory in choice_trajectories
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "task": records[0].benchmark,
        "record_count": len(records),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_batches": dict(Counter(record.metadata["source_batch"] for record in records)),
        "positive_labels": sum(record.label for record in records),
        "complete_trajectory_count": sum(trajectory.complete for trajectory in trajectories),
        "pairwise_choice_count": len(choices),
        "pairwise_positive_labels": sum(record.label for record in choices),
        "complete_choice_trajectory_count": sum(
            trajectory.complete for trajectory in choice_trajectories
        ),
        "capture_type": "real_historical_outcome_retrospective_replay",
        "hidden_outcome_contract": True,
        "confirmation_status": "development_only_source_inspected_before_protocol_lock",
        "warning": (
            "This dataset validates a real-outcome replay path but is too small and was "
            "inspected before construction; it is not untouched confirmatory evidence."
        ),
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
