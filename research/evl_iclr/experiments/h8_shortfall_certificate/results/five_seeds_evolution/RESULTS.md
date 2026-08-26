# H8 Hoeffding-Certified Utility Shortfall

The certificate controls normalized continuous utility shortfall, not binary failure probability.

## Fixed source-decomposed harness

| seed | coverage | selected gain | population net gain | mean shortfall | binary risk | acceptance UCB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0440 | 0.0251 | 0.000563 | 0.0136 | 0.2409 | 0.1760 |
| 7 | 0.0328 | 0.0197 | 0.000434 | 0.0155 | 0.2745 | 0.1691 |
| 13 | 0.0000 | 0.0000 | -0.001376 | 0.0000 | 0.0000 | 0.0000 |
| 21 | 0.0286 | 0.0317 | 0.000543 | 0.0087 | 0.2135 | 0.1733 |
| 42 | 0.0398 | 0.0292 | 0.000855 | 0.0088 | 0.2742 | 0.1714 |

## Hoeffding-certified recursive harness evolution

| seed | selected harness | accepted mutations | coverage | selected gain | population net gain | mean shortfall | binary risk | acceptance UCB |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `task_sparse_continuous_mean` | `task_sparse_continuous_mean` | 0.0440 | 0.0207 | 0.000374 | 0.0170 | 0.2701 | 0.1993 |
| 7 | `task_sparse_continuous_sd_worth` | `task_sparse_continuous_mean,task_sparse_continuous_sd_worth` | 0.0328 | 0.0197 | 0.000434 | 0.0155 | 0.2745 | 0.1922 |
| 13 | `base_mean` | `` | 0.0000 | 0.0000 | 0.000000 | 0.0000 | 0.0000 | 0.0000 |
| 21 | `task_sparse_continuous_sd_worth` | `task_sparse_continuous_mean,task_sparse_continuous_sd_worth` | 0.0286 | 0.0317 | 0.000543 | 0.0087 | 0.2135 | 0.1966 |
| 42 | `task_sparse_continuous_sd_worth` | `task_sparse_continuous_sd_worth` | 0.0398 | 0.0292 | 0.000855 | 0.0088 | 0.2742 | 0.1941 |

## Confirmatory audit

- `fixed_nonzero_coverage_seeds`: `4`
- `fixed_positive_selected_gain_seeds`: `4`
- `fixed_positive_population_net_gain_seeds`: `4`
- `fixed_test_shortfall_at_most_0.20_seeds`: `5`
- `evolved_nonzero_coverage_seeds`: `4`
- `evolved_positive_population_net_gain_seeds`: `4`
- `evolved_test_shortfall_at_most_0.20_seeds`: `5`
- `bernstein_nonzero_coverage_seeds`: `0`
- `h8_fixed_confirmatory_pass`: `True`
- `h8_evolved_supportive_pass`: `True`

## Interpretation

Hoeffding certification is evaluated on the same bounded shortfall loss, score grid, data splits, and gain targets as the earlier empirical-Bernstein certificate. Any coverage improvement therefore comes from concentration-bound power rather than a changed target. Test-set shortfall is a held-out empirical stress test; the formal statement requires acceptance/deployment exchangeability.
