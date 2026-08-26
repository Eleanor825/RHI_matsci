# SD-Worth H1.1: Budget-Conditioned Net Gain

Seeds: `[1, 13, 21, 42]`; action budget: `0.1`; delta: `0.05`.

## Target sanity

| Seed | Split | Worthy rate | Success mismatch | Mean gain |
|---:|---|---:|---:|---:|
| 1 | train | 0.1000 | 0.2203 | -0.7465 |
| 1 | feedback | 0.0908 | 0.2231 | -0.7540 |
| 1 | acceptance | 0.1014 | 0.2280 | -0.7438 |
| 1 | test | 0.1118 | 0.2245 | -0.7297 |
| 13 | train | 0.0994 | 0.2249 | -0.2189 |
| 13 | feedback | 0.1055 | 0.2088 | -0.2070 |
| 13 | acceptance | 0.0938 | 0.2114 | -0.2082 |
| 13 | test | 0.1144 | 0.2220 | -0.1713 |
| 21 | train | 0.1001 | 0.2219 | -0.7454 |
| 21 | feedback | 0.0972 | 0.2188 | -0.7509 |
| 21 | acceptance | 0.0904 | 0.2344 | -0.7486 |
| 21 | test | 0.1031 | 0.2278 | -0.7365 |
| 42 | train | 0.1000 | 0.2245 | -0.7451 |
| 42 | feedback | 0.0818 | 0.2218 | -0.7642 |
| 42 | acceptance | 0.0994 | 0.2203 | -0.7465 |
| 42 | test | 0.1070 | 0.2294 | -0.7311 |

## Risk-controlled test results at alpha=0.1

| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| binary_budget_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0951 | 0.0204 | 0.8357 |
| sd_worth_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.1220 | 0.1232 | 0.8578 |
| static_reliability | 0.0000 | 0.0000 | 0.0000 | 0.2272 | 0.3241 | 0.8597 |
| task_conditioned_binary_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0934 | 0.0237 | 0.8268 |
| task_conditioned_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.1167 | 0.0750 | 0.8407 |

## Risk-controlled test results at alpha=0.2

| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| binary_budget_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0951 | 0.0204 | 0.8357 |
| sd_worth_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.1220 | 0.1232 | 0.8578 |
| static_reliability | 0.0000 | 0.0000 | 0.0000 | 0.2272 | 0.3241 | 0.8597 |
| task_conditioned_binary_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0934 | 0.0237 | 0.8268 |
| task_conditioned_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.1167 | 0.0750 | 0.8407 |

## Risk-controlled test results at alpha=0.3

| Method | Coverage | Selective risk | Net gain/action | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| binary_budget_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0951 | 0.0204 | 0.8357 |
| sd_worth_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.1220 | 0.1232 | 0.8578 |
| static_reliability | 0.0000 | 0.0000 | 0.0000 | 0.2272 | 0.3241 | 0.8597 |
| task_conditioned_binary_worthiness | 0.0000 | 0.0000 | 0.0000 | 0.0934 | 0.0237 | 0.8268 |
| task_conditioned_gain_distribution | 0.0000 | 0.0000 | 0.0000 | 0.1167 | 0.0750 | 0.8407 |
