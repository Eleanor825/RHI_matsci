# Outcome-Blind Selected-Batch Risk Audit

| experiment | method | selected | errors | empirical risk | Hoeffding UCB | certified |
|---|---|---:|---:|---:|---:|---|
| `h5_full_acceptance` | `learned_portfolio_kg_mean` | 254 | 2 | 0.0079 | 0.0847 | `True` |
| `h23_primary` | `source_voi_expected` | 396 | 0 | 0.0000 | 0.0615 | `True` |
| `h23_evolved` | `fixed_sequence_evolved` | 297 | 0 | 0.0000 | 0.0710 | `True` |

Scope: average conditional risk of each frozen, outcome-blind selected batch under independent bounded losses; not individual-action conditional coverage.
