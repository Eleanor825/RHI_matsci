# H21 Unique-Material Alpha-0.10 Confirmatory Results

| method | certified folds | executed | errors | pooled risk | risk UCB | population gain |
|---|---:|---:|---:|---:|---:|---:|
| `source_voi_expected` | 2/4 | 287 | 12 | 0.0418 | 0.0669 | +0.000772 |
| `no_source_voi_expected` | 1/4 | 192 | 9 | 0.0469 | 0.0804 | +0.000324 |
| `external_variance` | 0/4 | 0 | 0 | 0.0000 | 0.0000 | +0.000000 |

Source-ablation gain delta: `+0.000447`.
Confirmatory gate passed: `False`.

## Recursive Signal Evolution

Selected contracts by fold: `['source_voi_expected', 'no_op', 'no_op', 'no_op']`.
Pooled evolved test gain: `+0.000431`.
Pooled evolved risk: `0.0192`.
Evolution gate passed: `False`.
