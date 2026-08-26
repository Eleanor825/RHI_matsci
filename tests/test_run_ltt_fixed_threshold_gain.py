import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "sd_worth"
    / "run_ltt_fixed_threshold_gain.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("h4_ltt_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class State:
    group_id: str
    benchmark: str = "task"


def test_three_way_feedback_split_is_group_disjoint_and_complete():
    states = [State(f"g{index // 2}") for index in range(30)]

    partitions = MODULE._split_feedback_three(states, 9)

    group_sets = [{row.group_id for row in partition} for partition in partitions]
    assert all(partitions)
    assert not (group_sets[0] & group_sets[1])
    assert not (group_sets[0] & group_sets[2])
    assert not (group_sets[1] & group_sets[2])
    assert sum(len(partition) for partition in partitions) == len(states)


def test_threshold_family_is_train_derived_nonnegative_and_small():
    thresholds = MODULE._threshold_family([-1.0, -0.2, 0.1, 0.4, 0.9] * 20)

    assert thresholds == sorted(set(thresholds))
    assert thresholds[0] == 0.0
    assert all(value >= 0.0 for value in thresholds)
    assert len(thresholds) <= 6
