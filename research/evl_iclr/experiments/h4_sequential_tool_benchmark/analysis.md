# H4 Analysis

## Seed-7 sanity result

The observation channel restored identifiability: verification evidence reduced
Brier score from `0.0886` to `0.0455`, and SD-Worth reached nonzero exact
risk-controlled test coverage at `alpha=0.20` (`1.25%` coverage, `7.69%`
risk). The source calibrator assigned positive external-variance weight on
the unique-material and extreme-property tasks.

H4 nevertheless fails its net-gain criterion. SD-Worth selected positive mean
gain (`0.0662`) but the predeclared verification cost (`0.075`) made the
selected trajectory gain negative. Charging this cost to every evaluated
action is even more negative. The variance-free verification model also
selected negative mean gain, while the direct tool probability had better
Brier/AURC but selected lower-magnitude positive gains.

## Mechanistic diagnosis

1. A shared cross-task regression ensemble is a poor model of task-specific
   utility magnitude even though the task label is included as a feature.
2. Calling an expensive verification tool for every proposed action is not a
   realistic host policy. A runtime harness should use a free/base score to
   acquire costly evidence only for a sparse subset.
3. Success probability alone ranks correctness but not scientific opportunity
   value; this explains why direct tool probability calibrates well yet has
   weak selected gain.

H4 is retained as a negative sanity result. H4.1 tests the preregistered
mechanistic corrections rather than silently changing the H4 interpretation.

## H4.1 seed-7 result

Task-specific heads plus sparse acquisition produced one positive baseline:
the sparse variance-free verification model selected 19 actions at zero
empirical risk and positive population net gain (`4.18e-4`) at cost `0.002`.
Sparse SD-Worth did not improve this result and selected negative mean gain.

The failure is not explained by binary risk alone. Full SD-Worth selected
143 actions with only `8.39%` binary risk, yet mean gain was negative because
the few unworthy actions had much larger negative magnitude than the positive
actions' gains. This motivates H5's continuous-gain likelihood and bounded
utility-shortfall certificate.

## H5 seed-7 result

Continuous Gaussian gain calibration fixed the severity-ranking failure:
at binary `alpha=0.20`, continuous SD-Worth selected positive mean gain
(`0.0634`) with `3.64%` empirical risk, and all three tasks learned nonzero
internal/external variance weights. The sparse continuous variants achieved
positive population net gain under the binary certificate.

The bounded-shortfall certificate remained unstable under the held-out task
mixture, and global sparse acquisition produced different acquisition rates
across train/feedback/test. H5.1 therefore tests task-conditional acquisition
and simultaneous task-wise certificates.

## H5.1 seed-7 result

Task-conditional acquisition reduced continuous SD verification to `10.7%` of
test actions. Under the pooled exact certificate, it selected 44 actions with
zero empirical error and positive cost-adjusted population gain at cost
`0.002`; its net gain exceeded the task-conditional continuous-mean ablation.

The simultaneous task-wise certificate had zero coverage because splitting
the acceptance set and applying Bonferroni left insufficient samples per task.
H5.2 retains stratified acquisition but uses a pooled acceptance certificate,
with its exchangeability assumption stated explicitly.

