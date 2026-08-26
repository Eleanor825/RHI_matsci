# H5 Protocol: Continuous Source-Decomposed Gain Certification

Date locked: 2026-08-21

## Motivation

H4.1 showed that a low binary selective risk does not imply positive net
scientific gain: a small number of high-severity negative actions can dominate
many small positive actions. H5 replaces binary-only source calibration with a
continuous likelihood for budget-conditioned scientific gain and adds a
bounded utility-shortfall certificate.

## Continuous calibration model

For each task, fit on the feedback split:

```text
G | s,a ~ Normal(
  a_task * mean_replica_gain + b_task,
  sigma_task^2
    + lambda_internal_task * U_internal
    + lambda_external_task * U_external
)
```

with `a_task > 0` and all variance terms non-negative. Parameters minimize
regularized Gaussian negative log likelihood on continuous gain, not binary
cross entropy. The scalar harness output remains

```text
q(s,a) = P(G > 0 | pre-action information).
```

## Risk certificates

Two independently selected acceptance thresholds are reported:

1. exact Clopper-Pearson control of `P(G <= 0 | accepted)`;
2. empirical-Bernstein control of bounded utility shortfall
   `min(1, max(0, -G / scale_task))`.

The shortfall scale is fit from training gains only. Candidate coverage grids
and Bonferroni correction are fixed as in H4.

## Sparse acquisition

Use H4.1's feedback-selected base-score acquisition threshold and cost
sensitivity. The primary normalized verification cost remains `0.002`.

## Confirmatory success criteria

Across at least four of five seeds:

1. continuous SD-Worth has nonzero shortfall-risk-controlled test coverage;
2. selected test mean gain and population net gain are positive;
3. it exceeds variance-free continuous mean calibration in population net
   gain under the same acquisition budget;
4. it does not worsen Brier score by more than `0.01` relative to the strongest
   probability baseline;
5. at least one source-variance coefficient is nonzero on at least two tasks.

