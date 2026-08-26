# SD-Worth H1.1: Budget-Conditioned Net Gain

Seeds: `[7]`; action budget: `0.1`; delta: `0.05`.

## Target sanity

| Seed | Split | Worthy rate | Success mismatch | Mean gain |
|---:|---|---:|---:|---:|
| 7 | train | 0.1001 | 0.2221 | -0.7470 |
| 7 | feedback | 0.0944 | 0.2215 | -0.7518 |
| 7 | acceptance | 0.1043 | 0.2183 | -0.7431 |
| 7 | test | 0.1041 | 0.2268 | -0.7371 |

## Risk-controlled test results at alpha=0.1

| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| binary_budget_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0903 | 0.0133 | 0.8386 |
| sd_worth_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.0934 | 0.0617 | 0.8469 |
| static_reliability | 0.0000 | 0.0000 | 0.0000 | 0.2335 | 0.3581 | 0.8560 |
| task_conditioned_binary_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0889 | 0.0220 | 0.8303 |
| task_conditioned_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.0891 | 0.0143 | 0.8282 |

## Risk-controlled test results at alpha=0.2

| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| binary_budget_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0903 | 0.0133 | 0.8386 |
| sd_worth_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.0934 | 0.0617 | 0.8469 |
| static_reliability | 0.0000 | 0.0000 | 0.0000 | 0.2335 | 0.3581 | 0.8560 |
| task_conditioned_binary_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0889 | 0.0220 | 0.8303 |
| task_conditioned_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.0891 | 0.0143 | 0.8282 |

## Risk-controlled test results at alpha=0.3

| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| binary_budget_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0903 | 0.0133 | 0.8386 |
| sd_worth_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.0934 | 0.0617 | 0.8469 |
| static_reliability | 0.0000 | 0.0000 | 0.0000 | 0.2335 | 0.3581 | 0.8560 |
| task_conditioned_binary_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0889 | 0.0220 | 0.8303 |
| task_conditioned_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.0891 | 0.0143 | 0.8282 |
