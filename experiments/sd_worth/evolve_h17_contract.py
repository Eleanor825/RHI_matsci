from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_matsci.contract_evolution import ContractCandidate, evolve_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    candidates = [
        _candidate(root, "joint_expected", "joint", "source_voi_expected", verify=False),
        _candidate(root, "joint_probability", "joint", "source_voi_probability", verify=False),
        _candidate(root, "verify_expected", "verify_before_execute", "source_voi_expected", verify=True),
        _candidate(root, "verify_probability", "verify_before_execute", "source_voi_probability", verify=True),
    ]
    result = evolve_contract(candidates, required_certification_rate=1.0).to_json()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")


def _candidate(root, name, family, method, *, verify):
    certified = []
    gains = []
    for fold in range(5):
        if verify:
            path = root / f"fold{fold}_verify_before_execute_dev" / "result.json"
        elif fold == 0:
            path = root / "fold0_linear" / "result.json"
        else:
            path = root / f"fold{fold}_confirmatory" / "result.json"
        row = json.loads(path.read_text(encoding="utf-8"))["policies"][method]
        certified.append(bool(row["deployed"]))
        gains.append(float(row["deployed_test_population_gain"]))
    return ContractCandidate(name, family, method, tuple(certified), tuple(gains))


def _markdown(result):
    lines = [
        "# H17 RHI Contract Evolution",
        "",
        "The evolution benchmark consists of five consumed log-KVRH group folds. A candidate is deployable only if it passes the independent certificate on every evolution fold.",
        "",
        "| candidate | contract | head | certified folds | mean deployed gain | accepted |",
        "|---|---|---|---:|---:|---|",
    ]
    accepted = {row["candidate"]: row["accepted"] for row in result["trace"]}
    for row in result["candidates"]:
        lines.append(
            f"| `{row['name']}` | `{row['contract_family']}` | `{row['acquisition_head']}` | "
            f"{sum(row['fold_certified'])}/{len(row['fold_certified'])} | "
            f"{row['mean_deployed_gain']:+.6f} | `{accepted[row['name']]}` |"
        )
    lines.extend(
        [
            "",
            f"Selected contract: `{result['selected']['name']}`.",
            "",
            "This selection uses only the H17 evolution benchmark. H18 and subsequent datasets are external transfer evaluations and are not inputs to the evolution gate.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
