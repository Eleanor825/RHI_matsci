from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness_matsci.lfp_crossed_gain import (
    CrossedPairwiseGainModel,
    VIEW_NAMES,
    apply_source_fuser,
    fit_candidate_excluded_source_fuser,
)
from harness_matsci.llm_action_choice import PROMPT_VERSION
from harness_matsci.schema import ActionRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", default="gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra"
    )
    parser.add_argument("--replicas", type=int, default=12)
    args = parser.parse_args()
    models = tuple(value for value in args.models.split(",") if value)
    caches = {
        model: _load_cache(Path(args.cache_dir) / f"{model}.json", model)
        for model in models
    }
    trajectories = []
    seen = set()
    for fold_dir in sorted(Path(args.folds_root).glob("fold_*")):
        train = _load_records(fold_dir / "train.jsonl")
        test = _load_records(fold_dir / "test.jsonl")
        fuser = fit_candidate_excluded_source_fuser(train)
        model = CrossedPairwiseGainModel(replicas=args.replicas).fit(train)
        states = apply_source_fuser(model.predict(test), fuser)
        for record, state in zip(test, states):
            if record.record_id in seen:
                raise ValueError(f"duplicate held-out LFP record {record.record_id}")
            seen.add(record.record_id)
            proposal = "A" if state.proposal_sign > 0.0 else "B"
            internal = []
            for model_name in models:
                key = f"{PROMPT_VERSION}::{record.record_id}::replica-0"
                prediction = caches[model_name].get(key)
                if prediction is None:
                    raise ValueError(f"missing LLM cell {model_name}, {record.record_id}")
                p_proposal = float(prediction["p_a_better"])
                if proposal == "B":
                    p_proposal = 1.0 - p_proposal
                internal.append(
                    {
                        "model": model_name,
                        "p_proposed_action_better": p_proposal,
                        "assessment_confidence": float(prediction["assessment_confidence"]),
                        "original_choice": prediction["choice"],
                    }
                )
            hidden = record.metadata["hidden_outcome_for_evaluation_only"]
            realized_gain = state.actual_comparative_gain - 0.005
            trajectories.append(
                {
                    "schema_version": "matbot-real-outcome-tool-replay-v1",
                    "trajectory_id": f"lfp-tool-replay::{record.record_id}",
                    "record_id": record.record_id,
                    "benchmark": record.benchmark,
                    "fold": fold_dir.name,
                    "capture_mode": "controlled_historical_tool_replay",
                    "online": False,
                    "matbot_runtime": False,
                    "real_scientific_outcome": True,
                    "steps": [
                        {
                            "step": 0,
                            "visible_context": record.visible_context,
                            "evidence": list(record.evidence),
                            "proposed_action": {
                                "choice": proposal,
                                "description": f"Execute candidate {proposal} rather than the alternative.",
                            },
                            "internal_model_replicas": internal,
                            "external_tool_evidence": {
                                "source_names": list(VIEW_NAMES),
                                "bootstrap_gain_matrix": [list(row) for row in state.source_matrix],
                                "fused_mean_gain": state.crossed_mean_gain,
                                "internal_variance": state.crossed_internal_variance,
                                "external_variance": state.crossed_external_variance,
                                "interaction_variance": state.crossed_interaction_variance,
                                "tool_cost": 0.005,
                                "training_contract": "fold-train-only with candidate-excluded source reliability fitting",
                            },
                            "hidden_outcome_available_pre_action": False,
                        }
                    ],
                    "terminal_outcome_for_training_and_evaluation_only": {
                        "candidate_a_measurement": hidden["candidate_a_measurement"],
                        "candidate_b_measurement": hidden["candidate_b_measurement"],
                        "utility_difference_a_minus_b": hidden["utility_difference"],
                        "proposal_aligned_utility_difference": state.actual_comparative_gain,
                        "realized_net_gain": realized_gain,
                        "worthy": int(realized_gain > 0.0),
                    },
                }
            )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    lines = []
    for trajectory in sorted(trajectories, key=lambda row: row["record_id"]):
        line = json.dumps(trajectory, ensure_ascii=False, sort_keys=True)
        lines.append(line)
        digest.update((line + "\n").encode())
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "matbot-real-outcome-tool-replay-v1",
        "trajectory_count": len(lines),
        "distinct_record_count": len(seen),
        "models": list(models),
        "source_names": list(VIEW_NAMES),
        "real_scientific_outcomes": True,
        "online": False,
        "matbot_runtime": False,
        "claim_boundary": (
            "Outcome-linked controlled replay with real LFP measurements and train-only tools; "
            "not a live MatBot execution campaign."
        ),
        "sha256": digest.hexdigest(),
        "output": str(destination),
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


def _load_cache(path: Path, model: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model or payload.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"invalid LFP cache contract for {model}")
    return payload.get("predictions", {})


def _load_records(path: Path):
    return [
        ActionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
