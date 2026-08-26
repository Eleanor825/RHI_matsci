from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.lfp_crossed_gain import (
    CrossedPairwiseGainModel,
    apply_source_fuser,
    evaluate_calibrated_methods,
    fit_candidate_excluded_source_fuser,
)
from harness_matsci.schema import ActionRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicas", type=int, default=12)
    args = parser.parse_args()
    fold_dir = Path(args.fold_dir)
    splits = {
        split: _load(fold_dir / f"{split}.jsonl")
        for split in ("train", "feedback", "acceptance", "test")
    }
    fuser = fit_candidate_excluded_source_fuser(splits["train"])
    model = CrossedPairwiseGainModel(replicas=args.replicas).fit(splits["train"])
    states = {
        split: apply_source_fuser(model.predict(records), fuser)
        for split, records in splits.items()
        if split != "train"
    }
    evaluation = evaluate_calibrated_methods(
        feedback_states=states["feedback"],
        acceptance_states=states["acceptance"],
        test_states=states["test"],
    )
    payload = {
        "fold_dir": str(fold_dir),
        "replicas": args.replicas,
        "source_fuser": fuser.to_json(),
        "split_counts": {split: len(records) for split, records in splits.items()},
        "evaluation": evaluation,
        "raw_states": {
            split: [state.to_json() for state in values]
            for split, values in states.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {
        method: {
            "certificate": values["certificate"],
            "metrics": values["test_metrics"],
            "deployed": values["deployed"],
            "conformal_coverage": values["test_conformal_coverage"],
        }
        for method, values in evaluation["methods"].items()
    }
    print(json.dumps(compact, sort_keys=True))


def _load(path: Path) -> list[ActionRecord]:
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
