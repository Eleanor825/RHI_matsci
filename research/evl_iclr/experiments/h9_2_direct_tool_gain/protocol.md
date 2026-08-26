# H9.2 Protocol: Direct Tool-Margin Gain Calibration

Date locked: 2026-08-21

## Motivation

H9.1 produced strong held-out task-native tool ranking on unique and pairwise
materials tasks, but the subsequent generic gain-head ensemble diluted that
signal and the base-score acquisition policy could not identify the actions
that benefited from verification.

## Method correction

1. Deterministic composition, crystal, and SMILES descriptors are computed from
   pre-action visible text and included in the zero-cost base observation.
2. Verification replicas remain frozen task-structured train-only scientific
   surrogates.
3. The direct tool representation is:

```text
mu_tool   = mean_e predicted_task_native_margin_e
U_external = Var_e predicted_task_native_margin_e
U_internal = base gain-head replica variance
```

4. A task-conditional heteroscedastic Gaussian calibrator maps
   `(mu_tool, U_internal, U_external)` directly to the distribution of strict
   budget-conditioned net gain `G`; no second verification gain-head is used.
5. The primary method is task-specific sparse acquisition of the direct
   source-decomposed score, with normalized verification cost `0.002`.

The scalar output remains `P(G>0)`; no route is emitted.

## Invariants and pilot gate

Data splits, gain target, normalization, risk certificate, coverage grid,
bootstrap tool replicas, outcome-invariance audit, and held-out regimes are
unchanged. Seed 7 is a development pilot. The exact method proceeds to five
seeds only if it obtains nonzero exact-risk coverage, positive selected gain,
and positive population net gain on seed 7.
