# H5.2 Five-Seed Results

Primary method: `task_sparse_continuous_sd_worth`; pooled exact risk certificate at `alpha=0.20`; verification cost `0.002`.

| seed | coverage | risk | selected gain | acquisition | population net gain | delta vs no-source | shortfall coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0157 | 0.0204 | 0.0662 | 0.2695 | 0.000504 | -0.000034 | 0.0000 |
| 7 | 0.0141 | 0.0000 | 0.0630 | 0.1066 | 0.000677 | +0.000353 | 0.0000 |
| 13 | 0.0173 | 0.1111 | 0.3661 | 0.6881 | 0.004975 | +0.006351 | 0.0000 |
| 21 | 0.0087 | 0.0741 | 0.0651 | 0.1815 | 0.000202 | -0.000022 | 0.0000 |
| 42 | 0.0154 | 0.0417 | 0.0624 | 0.1545 | 0.000654 | +0.000117 | 0.0000 |

## Aggregate

- Mean coverage: `0.0143`.
- Mean selective risk: `0.0495`.
- Mean acquisition fraction: `0.2801`.
- Mean population net gain: `0.001402`.
- Mean paired gain over no-source ablation: `+0.001353`.
- 95% seed-bootstrap CI for paired gain: `(8.825314117915772e-07, 0.003874394495652676)`.
- Wilcoxon p-value: `0.3125`.

## Confirmatory audit

- `nonzero_coverage_seeds`: `5`
- `risk_at_most_0.20_seeds`: `5`
- `positive_population_net_gain_seeds`: `5`
- `beats_continuous_mean_seeds`: `3`
- `positive_mean_paired_difference`: `True`
- `source_nonzero_two_tasks_every_seed`: `True`
- `h5_2_confirmatory_pass`: `True`

## Interpretation

The preregistered H5.2 criteria pass: sparse source-decomposed calibration gives nonzero, risk-controlled, positive net gain in all five seeds and beats the no-source continuous ablation in three seeds with a positive mean difference. The paired improvement is not statistically decisive with only five split seeds, so the source-decomposition advantage remains a promising but not final claim. The stricter bounded-shortfall certificate has low or zero coverage and remains an open requirement.
