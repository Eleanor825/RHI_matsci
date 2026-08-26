# H7 Risk-Controlled Harness Evolution

| seed | selected harness | accepted mutations | coverage | risk | population net gain | acquisition |
|---:|---|---|---:|---:|---:|---:|
| 1 | `task_sparse_continuous_mean` | `sparse_tool_probability,task_sparse_continuous_mean` | 0.0154 | 0.0000 | 0.000538 | 0.2695 |
| 7 | `task_sparse_continuous_sd_worth` | `sparse_verification_mean,task_sparse_continuous_mean,task_sparse_continuous_sd_worth` | 0.0141 | 0.0000 | 0.000677 | 0.1066 |
| 13 | `sparse_verification_mean` | `sparse_verification_mean` | 0.0235 | 0.0137 | 0.013511 | 1.0000 |
| 21 | `task_sparse_continuous_mean` | `task_sparse_continuous_mean` | 0.0090 | 0.0714 | 0.000224 | 0.1815 |
| 42 | `sparse_tool_probability` | `sparse_verification_mean,sparse_tool_probability` | 0.0096 | 0.0667 | 0.000264 | 0.0344 |

## Aggregate

- Mean population net gain: `0.003043`.
- Median population net gain: `0.000538`.
- Mean coverage: `0.0143`.
- Mean selective risk: `0.0304`.
- Final harness frequencies: `{"sparse_tool_probability": 1, "sparse_verification_mean": 1, "task_sparse_continuous_mean": 2, "task_sparse_continuous_sd_worth": 1}`.

## Fixed candidates

| method | mean net gain | median net gain | positive seeds |
|---|---:|---:|---:|
| `base_mean` | 0.000000 | 0.000000 | 0/5 |
| `sparse_verification_mean` | 0.002805 | 0.000248 | 3/5 |
| `sparse_tool_probability` | -0.000446 | -0.000220 | 1/5 |
| `task_sparse_continuous_mean` | 0.000049 | 0.000324 | 4/5 |
| `task_sparse_continuous_sd_worth` | 0.001402 | 0.000654 | 5/5 |

## Confirmatory audit

- `nonzero_coverage_seeds`: `5`
- `risk_at_most_0.20_seeds`: `5`
- `positive_net_gain_seeds`: `5`
- `mean_net_gain_above_base`: `True`
- `median_net_gain_above_base`: `True`
- `h7_confirmatory_pass`: `True`

## Interpretation

H7 passes its preregistered criteria and selects different contracts across regimes, demonstrating harness-level rather than action-level evolution. Both H7 and the fixed source-decomposed harness are positive in all five seeds. H7 has higher mean net gain because it accepts the strong sparse-verification contract in seed 13, while the fixed source harness has a slightly higher median. This is evidence that acceptance-set evolution can exploit regime-specific contract differences, not yet a statistically significant universal advantage.
