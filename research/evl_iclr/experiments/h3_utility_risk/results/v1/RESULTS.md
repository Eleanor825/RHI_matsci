# H3 Utility-Shortfall Selective Risk

| Method | alpha | Coverage | Shortfall risk | Binary risk | Mean selected gain | Net gain/action | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_reliability | 0.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| static_reliability | 0.2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| static_reliability | 0.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| binary_budget_worthiness | 0.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| binary_budget_worthiness | 0.2 | 0.1001 | 0.0141 | 0.1714 | 0.0067 | 0.00337 | 1/5 |
| binary_budget_worthiness | 0.3 | 0.1001 | 0.0141 | 0.1714 | 0.0067 | 0.00337 | 1/5 |
| sd_worth_gain_distribution | 0.1 | 0.1001 | 0.0001 | 0.1780 | 0.0229 | 0.01144 | 1/5 |
| sd_worth_gain_distribution | 0.2 | 0.1001 | 0.0001 | 0.1780 | 0.0229 | 0.01144 | 1/5 |
| sd_worth_gain_distribution | 0.3 | 0.1001 | 0.0001 | 0.1780 | 0.0229 | 0.01144 | 1/5 |
| extra_trees_classifier | 0.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| extra_trees_classifier | 0.2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| extra_trees_classifier | 0.3 | 0.1001 | 0.0310 | 0.1684 | -0.0135 | -0.00677 | 0/5 |
| hist_gradient_boosting | 0.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| hist_gradient_boosting | 0.2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0/5 |
| hist_gradient_boosting | 0.3 | 0.1001 | 0.0301 | 0.1710 | -0.0116 | -0.00583 | 0/5 |
| extra_trees_gain_regression | 0.1 | 0.1001 | 0.0000 | 0.1784 | 0.0233 | 0.01164 | 1/5 |
| extra_trees_gain_regression | 0.2 | 0.1001 | 0.0000 | 0.1784 | 0.0233 | 0.01164 | 1/5 |
| extra_trees_gain_regression | 0.3 | 0.1001 | 0.0000 | 0.1784 | 0.0233 | 0.01164 | 1/5 |
