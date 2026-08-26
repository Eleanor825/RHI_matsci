import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "experiments/sd_worth/merge_llm_worthiness_caches.py"
SPEC = importlib.util.spec_from_file_location("merge_llm_worthiness_caches", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path, predictions):
    path.write_text(json.dumps({"model":"m","prompt_version":"p","reasoning_effort":"medium","predictions":predictions}))


def test_merges_identical_overlaps(tmp_path):
    first=tmp_path/"a.json"; second=tmp_path/"b.json"; output=tmp_path/"out.json"
    _write(first,{"a":{"p":0.2},"shared":{"p":0.4}})
    _write(second,{"b":{"p":0.3},"shared":{"p":0.4}})
    report=MODULE.merge_cache_files([first,second],output)
    assert report["prediction_count"]==3
    assert report["identical_duplicate_count"]==1


def test_rejects_conflicting_overlap(tmp_path):
    first=tmp_path/"a.json"; second=tmp_path/"b.json"
    _write(first,{"shared":{"p":0.4}}); _write(second,{"shared":{"p":0.5}})
    with pytest.raises(ValueError,match="conflicting prediction"):
        MODULE.merge_cache_files([first,second],tmp_path/"out.json")
