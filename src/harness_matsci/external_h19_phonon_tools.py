from __future__ import annotations

from .external_h18_dielectric_tools import ExternalH18DielectricEvidenceEnvironment
from .external_h19_phonon_pairwise import DATASET, TASK


class ExternalH19PhononEvidenceEnvironment(ExternalH18DielectricEvidenceEnvironment):
    dataset_name = DATASET
    task_name = TASK
    source_tool_name = "train_only_phonon_source_model"
