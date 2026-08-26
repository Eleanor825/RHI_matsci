from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.matbot_trajectory import (
    load_matbot_trajectories,
    trajectory_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    trajectories = load_matbot_trajectories(args.events)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row.to_json(), sort_keys=True) for row in trajectories)
        + ("\n" if trajectories else ""),
        encoding="utf-8",
    )
    summary = trajectory_summary(trajectories)
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
