# H13 Task Oracle Headroom Diagnostic

This is an outcome-aware diagnostic upper bound, not a deployable method.
Hidden outcomes are used only to determine whether positive-gain actions exist
under the locked train-only normalization and opportunity-cost definition.

| fold | task | positive fraction | maximum gain | oracle population net gain |
|---:|---|---:|---:|---:|
| 0 | `matbench_pairwise` | 0.0883 | +0.0999 | +0.004276 |
| 0 | `discover_unique` | 0.1044 | +0.0988 | +0.004739 |
| 0 | `extreme_properties` | 0.0400 | +0.0998 | +0.001361 |
| 1 | `matbench_pairwise` | 0.0833 | +0.0986 | +0.003962 |
| 1 | `discover_unique` | 0.0829 | +0.0994 | +0.003995 |
| 1 | `extreme_properties` | 0.1125 | +0.0998 | +0.005493 |
| 2 | `matbench_pairwise` | 0.0958 | +0.0986 | +0.005296 |
| 2 | `discover_unique` | 0.1007 | +0.0998 | +0.005026 |
| 2 | `extreme_properties` | 0.3250 | +0.0998 | +0.009299 |
| 3 | `matbench_pairwise` | 0.1149 | +0.0999 | +0.005981 |
| 3 | `discover_unique` | 0.1501 | +0.0983 | +0.007674 |
| 3 | `extreme_properties` | 0.0000 | -1.0622 | +0.000000 |
| 4 | `matbench_pairwise` | 0.1168 | +0.0991 | +0.005213 |
| 4 | `discover_unique` | 0.0561 | +0.1000 | +0.002587 |
| 4 | `extreme_properties` | 0.0975 | +0.0998 | +0.004835 |

## Aggregate

- `matbench_pairwise`: mean positive fraction `0.0998`, mean oracle population gain `+0.004946`, positive headroom in `5/5` folds.
- `discover_unique`: mean positive fraction `0.0988`, mean oracle population gain `+0.004804`, positive headroom in `5/5` folds.
- `extreme_properties`: mean positive fraction `0.1150`, mean oracle population gain `+0.004197`, positive headroom in `4/5` folds.

## Conclusion

- Unique discovery has positive oracle headroom in every fold, so H13's zero
  selected unique actions are a ranking/evidence failure rather than an empty
  utility target.
- Extreme-property discovery has positive headroom in four folds but one
  structurally empty held-out regime, so both ranking and regime support matter.
- H14 must improve task-specific evidence before changing acceptance power.
