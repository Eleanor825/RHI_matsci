# H8 Protocol: Hoeffding-Certified Utility Shortfall

Date locked: 2026-08-21

## Motivation

The existing empirical-Bernstein certificate is valid but dominated by its
`3 log(3/delta)/(n-1)` term at current acceptance-set sizes, yielding zero
coverage for the primary sparse harness. H8 tests a predeclared Hoeffding
upper confidence bound for the same bounded utility-shortfall loss.

For accepted losses `L_i in [0,1]`, certify

```text
E[L] <= mean(L) + sqrt(log(1/delta') / (2 n))
```

where `delta'` is Bonferroni-corrected across the finite coverage grid and,
for H7-style candidate selection, across harness contracts. This is
distribution-free under IID/exchangeability and does not use empirical
variance.

## Fixed loss and methods

The loss remains train-normalized scientific shortfall:

```text
L_shortfall = min(1, max(0, -G / train_only_scale_task))
```

No target, utility normalizer, gain model, acquisition policy, candidate
coverage grid, or test split changes from H5.2/H7.

Primary methods:

- `task_sparse_continuous_sd_worth`;
- `task_sparse_continuous_mean`;
- H7 evolved harness under family-wise Hoeffding certification.

## Confirmatory criteria

At shortfall risk `alpha=0.20`, the primary fixed source harness succeeds if
at least four of five seeds have nonzero test coverage, positive selected mean
gain, and positive population net gain. Empirical test shortfall is reported;
the formal coverage guarantee applies under acceptance/deployment
exchangeability, while held-out regimes remain empirical stress tests.

H8 must also report the existing empirical-Bernstein result unchanged to make
clear that any power gain comes from the alternative valid concentration
bound, not from changing the loss.

