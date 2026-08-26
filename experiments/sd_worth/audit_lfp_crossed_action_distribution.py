from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


FORBIDDEN_TARGET_FIELDS = (
    "primary_dcr",
    "cycle_600",
    "conductivity_ms_cm",
    "candidate_a_measurement",
    "candidate_b_measurement",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    result = audit(payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def audit(payload: dict[str, object]) -> dict[str, object]:
    cells = list(payload["cells"])
    record_count = int(payload["record_count"])
    models = tuple(payload["models"])
    replicas = int(payload["replicas_per_model"])
    views = tuple(payload["external_views"])
    expected_cells = record_count * len(models) * replicas * len(views)
    keys = [
        (cell["record_id"], cell["agent_replica_id"], cell["view_id"])
        for cell in cells
    ]
    per_record = Counter(cell["record_id"] for cell in cells)
    prompt_leaks = []
    batch_leaks = []
    for record_id, metadata_rows in payload["view_metadata"].items():
        target_batch = next(
            cell["source_batch"] for cell in cells if cell["record_id"] == record_id
        )
        for metadata in metadata_rows:
            visible = json.dumps(
                {
                    "visible_context": metadata["visible_context"],
                    "evidence": metadata["evidence"],
                },
                ensure_ascii=False,
            ).lower()
            for field in FORBIDDEN_TARGET_FIELDS:
                if field.lower() in visible:
                    prompt_leaks.append(
                        {"record_id": record_id, "view_id": metadata["view_id"], "field": field}
                    )
            if metadata["view_id"] == "tool::held_batch_analogs":
                for result in metadata["provenance"].get("results", []):
                    if target_batch in result["record_id"]:
                        batch_leaks.append(
                            {
                                "record_id": record_id,
                                "analog_record_id": result["record_id"],
                            }
                        )
    expected_per_record = len(models) * replicas * len(views)
    passed = all(
        (
            len(cells) == expected_cells,
            len(keys) == len(set(keys)),
            len(per_record) == record_count,
            all(count == expected_per_record for count in per_record.values()),
            not prompt_leaks,
            not batch_leaks,
        )
    )
    return {
        "passed": passed,
        "expected_cells": expected_cells,
        "observed_cells": len(cells),
        "unique_cells": len(set(keys)),
        "records": len(per_record),
        "expected_cells_per_record": expected_per_record,
        "per_record_cell_counts": dict(sorted(per_record.items())),
        "prompt_leaks": prompt_leaks,
        "held_batch_retrieval_leaks": batch_leaks,
    }


if __name__ == "__main__":
    main()
