# H14 Protocol Amendment 3

Date: 2026-08-22

This amendment was made after inspecting only fold 0, which was declared and
remains development-only, and before running any H14 method on external folds
1--4.

The original deployment rule searched multiple coverage prefixes on acceptance
and required both a selective-risk upper bound and a bounded population-gain
lower bound to pass independently in every task-fold. Fold-0 power analysis
showed that this rule rejects a frozen policy with 20 acceptance actions, zero
errors, and positive held-out gain. The failure was caused by spending error
probability across acceptance prefixes and by applying a range-dominated gain
bound to a low-coverage population, not by observed scientific failures.

The revised protocol separates the data roles:

1. feedback selects one task-conditional method, coverage, and numeric
   threshold;
2. acceptance evaluates that frozen policy exactly once;
3. the one-sided exact selective-risk bound uses `delta=0.05/3` across tasks;
4. a zero-selection policy is not certified;
5. test copies the accepted method, threshold, and tool contract unchanged.

Selective risk remains the finite-sample deployment guarantee at
`alpha=0.20`. Net gain remains the scientific objective and includes all tool
costs, but confirmatory evidence for gain is aggregated through the already
predeclared task-fold success counts and paired randomization comparisons,
rather than an underpowered per-fold bounded-mean gate.

This amendment does not change datasets, outer folds, target definitions,
costs, alpha, test metrics, baselines, or the requirement that each task obtain
positive nonzero-coverage gain in at least four of five outer folds. The
pre-amendment hash is retained in `protocol-lock-v3.sha256`.
