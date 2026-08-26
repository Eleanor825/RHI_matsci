from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=(
            "/Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks/"
            "cache/matbench_log_kvrh.json.bz2"
        ),
    )
    parser.add_argument("--output-root", default="benchmarks/external_h15_unique/raw")
    args = parser.parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "matbench_log_kvrh.json.bz2"
    shutil.copyfile(source, destination)
    manifest = {
        "snapshot_date": "2026-08-22",
        "dataset": "matbench_log_kvrh",
        "rows": 10987,
        "target": "log10(K_VRH)",
        "role": "powered_discover_unique_external_confirmation",
        "source": str(source),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "inspection_policy": (
            "Added after H14 fold-0 power analysis proved the 636-row unique task "
            "cannot satisfy the locked finite-sample risk certificate even under oracle ranking."
        ),
    }
    (root.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
