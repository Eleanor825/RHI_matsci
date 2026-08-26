from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.direct_judge import PROMPT_VERSION
from harness_matsci.direct_judge_subset import DirectJudgeSubsetConfig, evaluate_direct_judge_subset, split_records_file
from harness_matsci.io import write_jsonl
from harness_matsci.schema import ActionRecord


class DirectJudgeSubsetTests(unittest.TestCase):
    def test_split_and_evaluate_subset_with_cached_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw_path = directory / "raw.jsonl"
            split_path = directory / "records.jsonl"
            cache_path = directory / "scores.json"
            out_dir = directory / "out"
            records = [_record(index) for index in range(72)]
            write_jsonl(records, raw_path)

            summary = split_records_file(raw_path, split_path, seed=11)
            self.assertEqual(summary["splits"], {"test": 36, "validation": 36})

            split_records = [json.loads(line) for line in split_path.read_text().splitlines()]
            cache_path.write_text(
                json.dumps(
                    {
                        "model": "fake-gpt-5.5",
                        "prompt_version": PROMPT_VERSION,
                        "reasoning_effort": "xhigh",
                        "scores": {record["record_id"]: 0.8 if record["label"] else 0.2 for record in split_records},
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_direct_judge_subset(
                DirectJudgeSubsetConfig(
                    records_path=str(split_path),
                    judge_cache_path=str(cache_path),
                    out_dir=str(out_dir),
                    epochs=8,
                    rhi_iterations=1,
                )
            )
            self.assertIn("llm_direct_judge", report["method_summary"])
            self.assertIn("scivoi_rhi", report["method_summary"])
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "README.md").exists())


def _record(index: int) -> ActionRecord:
    task = ("matbench_pairwise", "discover_unique", "extreme_properties")[index % 3]
    label = 1 if index % 4 in {0, 1} else 0
    return ActionRecord(
        record_id=f"r-{index:03d}",
        benchmark=task,
        split="unspecified",
        visible_context=f"Visible scientific context {index}",
        candidate_action=f"Execute action {index}",
        action_type="choose" if task == "matbench_pairwise" else "recommend",
        evidence=["visible surrogate evidence"],
        features={
            "verbal_confidence": 0.75 if label else 0.25,
            "evidence_support": 0.8 if label else 0.2,
            "evidence_conflict": 0.2 if label else 0.8,
            "source_reliability": 0.7,
            "tool_agreement": 0.7 if label else 0.3,
            "model_disagreement": 0.2 if label else 0.6,
            "perturbation_stability": 0.7 if label else 0.3,
            "ood_score": 0.2 if label else 0.6,
            "cost": 0.3 + 0.01 * (index % 10),
            "reversibility": 0.6,
            "action_complexity": 0.4,
            "evidence_count": 0.5,
        },
        label=label,
        utility=float(label) + 0.01 * index,
        metadata={"group_id": f"{task}::{index % 6}"},
    )


if __name__ == "__main__":
    unittest.main()
