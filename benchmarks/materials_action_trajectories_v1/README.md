# Materials Action-Trajectory Benchmark v1

This package contains the complete offline benchmark used by the current
materials uncertainty-harness experiments. It covers the three main tasks:

- `matbench_pairwise`: 8,000 real Matbench elasticity A/B preference actions.
- `discover_unique`: 10,987 DiSCoVeR-style unique-material screening actions.
- `extreme_properties`: 2,000 RL-CC extreme-property actions.

## Data unit

The supervised unit is an **action-level decision**. Each action has a complete
record containing the visible decision-time input, internal/external uncertainty
signals, the hidden benchmark outcome, utility, label provenance, and raw
source record. The same action is also placed in a grouped trajectory view,
ordered by `action_index` within its benchmark regime.

This is deliberately explicit because the current data are historical offline
reconstructions, not live MatBot logs. `trajectory_complete=true` means the
benchmark record contains its complete reconstructed context and outcome; it
does not claim that it came from an online multi-step MatBot execution.

## Files

- `actions.jsonl.gz`: one line per action-level decision unit (`20,987` lines).
- `trajectories.jsonl.gz`: grouped benchmark-regime trajectories (`45` lines).
- `manifest.json`: counts, label sources, leakage contract, and SHA-256 hashes.

Each action separates:

- `model_input`: legal at decision time; use this for gate training/inference.
- `outcome`: label and utility used for training/evaluation feedback.
- `raw_record`: full provenance-preserving source record for audit only.

Do not feed `outcome` or `raw_record` into the runtime gate. The benchmark
labels are benchmark-derived action-worthiness proxies and should be reported
separately from expert or online MatBot labels.

## Rebuild

```bash
PYTHONPATH=src python3 -m harness_matsci build-trajectory-benchmark \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --tasks matbench_pairwise,discover_unique,extreme_properties \
  --out-dir benchmarks/materials_action_trajectories_v1
```

## Self-evolution connection

The `rhi` implementation now uses a signal-aware proposer by default. At each
round it summarizes feedback-split error associations for internal signals
(`verbal_confidence`, perturbation stability, self-consistency, disagreement)
and external signals (evidence, source reliability, tool agreement, OOD), then
mutates the harness contract and hops. Candidate revisions are still trained
on the training split and accepted only on the held-out acceptance shard;
test data remain untouched until final evaluation.
