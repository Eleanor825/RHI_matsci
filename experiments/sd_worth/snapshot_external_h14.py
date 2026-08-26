from __future__ import annotations

import hashlib
import json
import shutil
from importlib.util import find_spec
from pathlib import Path


DATASETS = {
    "matbench_dielectric": {
        "filename": "matbench_dielectric.json.gz",
        "rows": 4764,
        "target": "n",
        "role": "pairwise_preference_external_confirmation",
    },
    "matbench_jdft2d": {
        "filename": "matbench_jdft2d.json.gz",
        "rows": 636,
        "target": "exfoliation_en",
        "role": "unique_discovery_external_confirmation",
    },
    "matbench_phonons": {
        "filename": "matbench_phonons.json.gz",
        "rows": 1265,
        "target": "last phdos peak",
        "role": "extreme_property_external_confirmation",
    },
}


def main() -> None:
    spec = find_spec("matminer.datasets")
    if spec is None or spec.origin is None:
        raise RuntimeError("matminer.datasets is unavailable")
    source_root = Path(spec.origin).parent
    output_root = Path("benchmarks/external_h14/raw")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "snapshot_date": "2026-08-22",
        "inspection_policy": (
            "Schemas and row counts were inspected before protocol lock; target "
            "distributions and method performance were not used to select datasets."
        ),
        "datasets": {},
    }
    for name, metadata in DATASETS.items():
        source = source_root / metadata["filename"]
        if not source.exists():
            raise FileNotFoundError(source)
        destination = output_root / metadata["filename"]
        shutil.copyfile(source, destination)
        manifest["datasets"][name] = {
            **metadata,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_url": f"https://ml.materialsproject.org/projects/{metadata['filename']}",
        }
    Path("benchmarks/external_h14/manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
