# Full Same-Evidence LLM Comparator Protocol

Frozen on August 23, 2026 after the balanced 485-record subset analysis and
before generating LLM predictions for the remaining acceptance records.

## Purpose

The 485-record balanced subset showed higher strict trajectory gain for the
frozen numerical harness than every LLM judge, but the paired one-sided
randomization test did not reach 5% significance. This power extension uses
all 2,538 records in the already frozen H5 acceptance set. It does not tune or
refit the numerical harness.

## Frozen comparators

The 485-record development comparison freezes two LLM comparators:

1. `screening::ensemble::platt_p`, the highest-gain LLM method regardless of
   certification;
2. `verification::ensemble::platt_p`, the highest-gain LLM method satisfying
   the `alpha=0.10` acceptance certificate on the subset.

The ensemble is the arithmetic mean of `gpt-5.6-sol`, `gpt-5.6-luna`, and
`gpt-5.6-terra`. Task-specific Platt calibrators remain fit only on the frozen
600-record feedback subset. No full-acceptance outcome is used for prompt,
model, stage, calibration, or comparator selection.

## Evaluation

- Same records, same visible evidence, same strict gain, and the same 10%
  action budget are used for the numerical harness and LLM judges.
- The primary effect is paired total strict trajectory gain of frozen
  `learned_portfolio_kg_mean` minus
  `screening::ensemble::platt_p`.
- The safety-qualified secondary effect replaces the comparator with
  `verification::ensemble::platt_p`.
- Report a benchmark-stratified paired bootstrap 95% interval and a one-sided
  paired sign-flip randomization p-value with seed `1729` and 100,000 draws.
- Superiority requires positive gain difference, bootstrap lower bound above
  zero, and randomization `p < 0.05`.
- Risk superiority is reported separately using selective risk and the
  one-sided 95% Clopper--Pearson upper bound; it is not folded into the gain
  significance test.

## Frozen hashes

- full acceptance: `cdd3140719bb381ead240f5eb56ea5980b725596b494d3857230873fb62cd6cb`
- feedback subset: `dfe2ec534609fe256e3c202d665d0e7624f0c284cef9514a8c053dd830f89b82`
- LLM runner: `46e3fb4dec1cdcaf684bbb052942209ea700542a55f05e862acba0d151336891`
- LLM evaluator with audit-only record identifiers: `a9e18e8dd58934b57b456a720ac4fd5428f0dcf9731912477b5f79128cd1d855`
