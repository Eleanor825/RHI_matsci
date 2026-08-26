import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "sd_worth"
    / "finalize_llm_worthiness_cache.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_cache", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _cache(path, predictions):
    path.write_text(
        json.dumps(
            {
                "model": "m1",
                "prompt_version": "strict",
                "reasoning_effort": "medium",
                "predictions": predictions,
            }
        )
    )


def test_finalizer_prunes_extras_and_requires_complete_expected_grid(tmp_path):
    trajectories = tmp_path / "rows.jsonl"
    trajectories.write_text(
        "\n".join(
            json.dumps({"record_id": record_id}) for record_id in ("a", "b")
        )
        + "\n"
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _cache(first, {"a::base": {"p": 1}, "extra::base": {"p": 0}})
    _cache(second, {"b::base": {"p": 2}})
    output = tmp_path / "final.json"

    report = MODULE.finalize_cache(
        trajectories=trajectories,
        inputs=[first, second],
        output=output,
        model="m1",
        prompt_version="strict",
        reasoning_effort="medium",
        view_id="base",
    )

    assert report["prediction_count"] == 2
    assert report["pruned_extra_count"] == 1
    assert set(json.loads(output.read_text())["predictions"]) == {"a::base", "b::base"}


def test_finalizer_rejects_conflicting_duplicates(tmp_path):
    trajectories = tmp_path / "rows.jsonl"
    trajectories.write_text(json.dumps({"record_id": "a"}) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _cache(first, {"a::base": {"p": 1}})
    _cache(second, {"a::base": {"p": 2}})

    with pytest.raises(ValueError, match="conflicting"):
        MODULE.finalize_cache(
            trajectories=trajectories,
            inputs=[first, second],
            output=tmp_path / "final.json",
            model="m1",
            prompt_version="strict",
            reasoning_effort="medium",
            view_id="base",
        )
