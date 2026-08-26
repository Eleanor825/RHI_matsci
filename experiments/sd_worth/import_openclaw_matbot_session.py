from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.openclaw_trajectory import load_openclaw_matbot_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    trajectory = load_openclaw_matbot_trajectory(
        args.session,
        system=args.system,
        stage=args.stage,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trajectory.to_json(), sort_keys=True) + "\n")
    step = trajectory.steps[0]
    summary = {
        "trajectory_count": 1,
        "step_count": 1,
        "capture_mode": trajectory.capture_mode,
        "complete": trajectory.complete,
        "agent_backend": step.agent_backend,
        "tool_call_count": len(step.tool_trace),
        "tool_error_count": sum(row["is_error"] for row in step.tool_trace),
        "action_id": step.action_id,
        "uncertainty_output": step.uncertainty_output,
    }
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
