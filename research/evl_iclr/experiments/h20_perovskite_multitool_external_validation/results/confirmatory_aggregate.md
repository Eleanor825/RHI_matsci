# H20 Perovskite Multi-Tool Confirmatory Aggregate

| method | certified folds | mean test gain | non-HF folds | selected subsets |
|---|---:|---:|---:|---|
| `source_multitool_expected` | 5/5 | +0.001390 | 0 | high_fidelity, high_fidelity, high_fidelity, high_fidelity, high_fidelity |
| `no_source_multitool_expected` | 5/5 | +0.001410 | 0 | high_fidelity, high_fidelity, high_fidelity, high_fidelity, high_fidelity |
| `fixed_high_fidelity_expected` | 5/5 | +0.001390 | 0 | high_fidelity, high_fidelity, high_fidelity, high_fidelity, high_fidelity |
| `static_reliability_high_fidelity` | 0/5 | +0.000000 | 0 | none, none, none, none, none |

## Frozen criteria

- `deploys_at_least_four_of_five`: `True`
- `positive_mean_gain`: `True`
- `beats_no_source`: `False`
- `beats_fixed_high_fidelity`: `False`
- `uses_non_high_fidelity_on_two_folds`: `False`
- `observed_test_risk_within_alpha`: `True`

Overall H20 confirmatory pass: `False`.
