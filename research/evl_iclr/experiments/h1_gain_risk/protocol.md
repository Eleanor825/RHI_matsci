# H1 Protocol — Strict Net Gain and Risk-Controlled Worthiness

Status: LOCKED BEFORE EXECUTION  
Locked date: 2026-08-21

## Hypothesis

A model trained on a train-only normalized continuous net scientific gain target and calibrated to predict `P(gain > 0)` will improve held-out-regime normalized net gain at a fixed selective-risk limit relative to the current static reliability classifier.

## Strict Target

For action `i` in task `t`:

```text
gain_i = success_i * normalized_utility_i
         - lambda_cost * normalized_cost_i
         - lambda_failure * (1 - success_i) * failure_harm_i
```

`normalized_utility_i` is fit only on the training split for task `t` using an empirical-CDF transform. `failure_harm_i` combines normalized cost and irreversibility. The binary worthiness target is `1[gain_i > 0]`, but the model must retain the continuous gain target.

## Risk Guarantee

Choose a probability threshold using only the calibration/acceptance split. For each candidate selected prefix, compute a one-sided exact binomial upper confidence bound on negative-worthiness risk. Use a union-adjusted confidence level across all candidate prefixes. Select the largest-coverage prefix whose upper bound is at most `alpha`.

## Evaluation

- Split: existing grouped held-out-regime four-way split.
- Tasks: Matbench pairwise, unique-material discovery, extreme-property discovery.
- Baseline: static reliability under the identical split and action budget.
- Primary: normalized net gain subject to selective risk upper bound `<= alpha`.
- Secondary: empirical selective risk, coverage, Brier, ECE, worst-regime risk.
- Test split is used once after model and threshold selection.

## Confirmatory Prediction

SD-Worth increases mean held-out net gain without violating the calibration-side finite-sample risk criterion. If it does not beat static reliability, the hypothesis is refuted and the gain target/model must be revised before adding complexity.

## Non-Claims

This experiment does not establish the internal/external decomposition or real MatBot performance.
