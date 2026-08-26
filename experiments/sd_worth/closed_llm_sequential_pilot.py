from __future__ import annotations

import argparse
import json
import os

from harness_matsci.closed_llm_sequential_pilot import DEFAULT_MODELS, run_closed_llm_sequential_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--n-per-task", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--selection-mode",
        choices=("deterministic", "label_balanced_diagnostic"),
        default="deterministic",
    )
    args = parser.parse_args()
    report = run_closed_llm_sequential_pilot(
        args.raw_data_dir,
        args.out_dir,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
        models=tuple(model for model in args.models.split(",") if model),
        n_per_task=args.n_per_task,
        workers=args.workers,
        selection_mode=args.selection_mode,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
