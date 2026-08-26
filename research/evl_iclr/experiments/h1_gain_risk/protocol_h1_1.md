# H1.1 Protocol — Budget-Conditioned Net Gain

Status: LOCKED BEFORE EXECUTION  
Locked date: 2026-08-21

## Motivation

H1 was refuted because positive net gain exactly matched success. Scientific action worthiness must include the opportunity cost of allocating a limited experiment/simulation budget.

## Target

First compute outcome-conditioned base gain:

```text
base_gain_i = success_i * normalized_utility_i
              - lambda_cost * normalized_cost_i
              - lambda_failure * (1 - success_i) * failure_harm_i
```

For each task, fit a train-only reservation value at action budget `B`:

```text
reservation_t(B) = quantile_{1-B}({base_gain_i : i in train task t})
```

The budget-conditioned net gain is:

```text
gain_i(B) = base_gain_i - reservation_t(B)
worthiness_i(B) = 1[gain_i(B) > 0]
```

This is the marginal value of spending one limited scientific action slot on `i` instead of the train-distribution reservation action.

## Risk Control

Predeclare coverage candidates `{0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}`. The score determines each prefix without labels. On the independent acceptance split, compute a one-sided exact binomial upper bound with Bonferroni correction across these eight candidates. Select the largest coverage whose upper risk bound is at most `alpha`.

## Confirmatory Comparison

Compare static success reliability, binary budget-conditioned worthiness, and continuous gain-distribution SD-Worth on identical held-out regimes. Report risk-controlled coverage/net gain at `alpha in {0.1, 0.2, 0.3}` without choosing alpha from test results.
