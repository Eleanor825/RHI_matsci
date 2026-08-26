# H13 Powered-Acceptance Confirmatory Results

Fold 0 is excluded. These rows are untouched regime folds 1--4.

| fold | policy | coverage | risk | selected gain | population net gain | delta vs no-source | tasks |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `no_op` | 0.0000 | 0.0000 | 0.0000 | +0.000000 | +0.000000 | `` |
| 2 | `activate_frozen_contract` | 0.0092 | 0.0588 | 0.0711 | +0.000386 | +0.000000 | `matbench_pairwise` |
| 3 | `activate_frozen_contract` | 0.0245 | 0.1149 | 0.0520 | +0.000803 | +0.000000 | `matbench_pairwise` |
| 4 | `activate_frozen_contract` | 0.0127 | 0.0833 | 0.0611 | +0.000515 | +0.000028 | `matbench_pairwise` |

## Aggregate

- Active folds: `3/4`.
- Positive-gain folds: `3/4`.
- Risk at most 0.20: `4/4`.
- Mean coverage: `0.011613`.
- Mean population net gain: `+0.000426`.
- Mean no-source gain: `+0.000419`.
- Mean source advantage: `+0.000007`; wins `1/4`.
- Tasks selected: `matbench_pairwise`.
- Locked confirmatory pass: `False`.

## Static baselines

- `base_mean` mean population net gain: `-0.000138`.
- `base_ensemble_lcb` mean population net gain: `-0.000138`.
- `base_verbal_confidence` mean population net gain: `+0.000000`.
- `base_evidence_heuristic` mean population net gain: `+0.000000`.
