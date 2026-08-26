# H3 Protocol — Utility-Shortfall Selective Risk

Status: LOCKED BEFORE EXECUTION  
Locked date: 2026-08-21

## Motivation

H1 treats every action below the budget reservation value as an equally severe error. This makes a marginally sub-reservation but scientifically useful action indistinguishable from an expensive total failure and yields zero coverage even for nonlinear models. Scientific risk should preserve the magnitude of lost value.

## Gain and Shortfall

Retain the locked train-only budget-conditioned gain:

```text
gain_i = base_gain_i - reservation_task(B)
```

For each task, fit a train-only lower reference `q05_task`, the fifth percentile of base gain. Define bounded scientific shortfall:

```text
scale_task = max(epsilon, reservation_task(B) - q05_task)
loss_i = clip(max(0, -gain_i) / scale_task, 0, 1)
```

`loss=0` means the action meets or exceeds the opportunity-cost reservation; a small miss has small loss; a severe failure approaches one.

## Selective Risk Guarantee

For each predeclared score-ranked coverage in `{0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}`, estimate mean shortfall on the independent acceptance split. Use a simultaneous empirical-Bernstein upper confidence bound for bounded losses with Bonferroni correction over the eight coverages. Select the largest coverage whose upper bound is no greater than `alpha`.

Report `alpha in {0.10, 0.20, 0.30}` without choosing alpha from test results. The guarantee assumes exchangeability between acceptance and deployment actions. Held-out-regime performance remains an empirical transfer test.

## Models

Evaluate locked scores without changing their training:

- static success reliability;
- binary budget-conditioned worthiness;
- SD-Worth gain distribution;
- ExtraTrees worthiness classifier;
- histogram gradient boosting worthiness classifier;
- ExtraTrees continuous gain regression.

## Primary Success Criterion

At `alpha=0.20`, at least one utility-aware method must obtain non-zero coverage, positive test budget-conditioned net gain per action, and higher gain than static reliability. Binary selective risk remains a reported secondary stress test and is not redefined.

This protocol changes the scientific loss, not the score labels or test split.
