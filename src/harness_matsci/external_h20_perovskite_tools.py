from __future__ import annotations

from .external_h18_dielectric_tools import ExternalH18DielectricEvidenceEnvironment
from .external_h20_perovskite_pairwise import (
    DATASET,
    TASK,
    load_external_h20_materials,
)


class ExternalH20PerovskiteEvidenceEnvironment(
    ExternalH18DielectricEvidenceEnvironment
):
    dataset_name = DATASET
    task_name = TASK
    source_tool_name = "train_only_perovskite_source_model"

    @classmethod
    def load_materials(cls, snapshot_root):
        return load_external_h20_materials(snapshot_root)
