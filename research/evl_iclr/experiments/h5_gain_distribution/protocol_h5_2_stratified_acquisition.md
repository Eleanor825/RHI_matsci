# H5.2 Protocol: Stratified Acquisition with Pooled Risk Certification

Date locked: 2026-08-21

## Motivation

H5.1's simultaneous task-wise certificates selected no actions because each
task-specific acceptance set was too small after Bonferroni correction. The
prediction mechanism was not degenerate: task-conditional acquisition plus a
pooled certificate selected zero-error, positive-gain actions in seed 7.

H5.2 therefore separates two roles:

- task-conditional feedback thresholds allocate the verification budget;
- a pooled, untouched acceptance set certifies the final scalar worthiness
  score under the deployment task mixture.

The pooled guarantee assumes exchangeability of the acceptance and deployment
mixtures. Held-out tasks or shifted mixtures remain empirical stress tests and
are not described as distribution-free guarantees.

## Primary method

`task_sparse_continuous_sd_worth`:

1. task-specific gain regressors;
2. complete 3-model x 3-tool replica matrices;
3. continuous source-decomposed Gaussian gain calibration;
4. feedback-selected task-specific base thresholds for sparse verification;
5. pooled exact binary and bounded-shortfall acceptance certificates;
6. scalar output `q = P(G > 0 | observations)`.

## Baselines

- `base_mean` static reliability;
- `tool_probability` direct tool-derived success judge;
- `sparse_verification_mean` binary mean-gain calibration;
- `task_sparse_continuous_mean` continuous gain without source variances;
- full-acquisition `continuous_sd_worth`.

## Confirmatory five-seed criteria

At `alpha=0.20` and verification cost `0.002`:

1. nonzero exact-risk-controlled test coverage in at least four seeds;
2. empirical test selective risk at most `0.20` in at least four seeds;
3. positive population net gain in at least four seeds;
4. higher population net gain than `task_sparse_continuous_mean` in at least
   three seeds and positive mean paired difference;
5. source-variance weights nonzero on at least two tasks in every seed.

Bounded-shortfall coverage is reported as a stricter secondary endpoint; zero
coverage is not hidden or reinterpreted as success.

