# H10 Real Scientific Tool Results

Primary: `task_sparse_tool_margin_continuous_sd_worth`; exact binary selective risk at `alpha=0.20`; normalized verification cost `0.002`.

| seed | audit | coverage | risk | selected gain | acquisition | population net gain | delta vs no-source | tasks selected |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | True | 0.0000 | 0.0000 | 0.0000 | 0.1792 | -0.000358 | +0.000000 | `` |
| 7 | True | 0.0151 | 0.0851 | 0.0552 | 0.1895 | 0.000454 | +0.000000 | `matbench_pairwise` |
| 13 | True | 0.0000 | 0.0000 | 0.0000 | 0.2046 | -0.000409 | +0.000000 | `` |
| 21 | True | 0.0000 | 0.0000 | 0.0000 | 0.1005 | -0.000201 | +0.000000 | `` |
| 42 | True | 0.0000 | 0.0000 | 0.0000 | 0.0793 | -0.000159 | +0.000000 | `` |

## Aggregate

- Mean coverage: `0.003020`; median `0.000000`.
- Mean selective risk: `0.017021`; median `0.000000`.
- Mean selected mean gain: `0.011041`; median `0.000000`.
- Mean acquisition fraction: `0.150659`; median `0.179248`.
- Mean population net gain: `-0.000135`; median `-0.000201`.
- Mean delta vs no source: `0.000000`; median `0.000000`.
- Mean delta vs static: `-0.000135`; median `-0.000201`.

## Confirmatory audit

- `audit_pass_seeds`: `5`
- `nonzero_coverage_seeds`: `1`
- `risk_at_most_0.20_seeds`: `5`
- `positive_selected_gain_seeds`: `1`
- `positive_population_net_gain_seeds`: `1`
- `beats_no_source_seeds`: `0`
- `positive_mean_delta_vs_no_source`: `False`
- `positive_mean_delta_vs_static`: `False`
- `h10_confirmatory_pass`: `False`

## Paired inference

- `source_bootstrap_mean_ci_95`: `(0.0, 0.0)`
- `source_wilcoxon`: `{'statistic': 0.0, 'pvalue': 1.0}`
- `static_bootstrap_mean_ci_95`: `(-0.0003489881143591391, 0.00016923172875281628)`
- `static_wilcoxon`: `{'statistic': 5.0, 'pvalue': 0.625}`
