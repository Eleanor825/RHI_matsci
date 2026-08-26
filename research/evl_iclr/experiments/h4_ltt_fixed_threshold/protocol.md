# H4 Fixed-Threshold Selective-Gain Protocol

This protocol addresses the action-level guarantee boundary identified during
the H3 method audit. H3 remains frozen and is not retroactively changed.

## Target

Use the same strict realized gain

`G = S U - 0.15 C - 0.25 (1-S) H`.

The score remains the numerical one-sided factorized gain lower bound. The
harness does not output a route.

## Learn-then-Test policy

1. Fit the score model using training groups only.
2. Construct a finite threshold family using training scores only.
3. Evaluate every frozen threshold on feedback outcomes.
4. Apply a Bonferroni-corrected one-sided exact risk bound across the complete
   threshold family.
5. Among simultaneously certified thresholds, choose the one with the largest
   feedback population net gain.
6. Freeze that scalar threshold before a new acceptance set is generated.

An action is eligible independently when its score is at least the frozen
threshold. If a deployment batch has more eligible actions than the hard
budget, uniformly thin eligible actions using an outcome-independent seeded
hash. This preserves a numerical harness/downstream allocator separation and
avoids claiming that a batch top-k Clopper--Pearson calculation is an
assumption-free action-level theorem.

## Confirmatory guarantee

On a newly generated group-disjoint acceptance set, the frozen policy must
have positive selected total gain and a one-sided 95% Clopper--Pearson upper
bound no greater than `alpha=0.10`. The guarantee is explicitly conditional on
i.i.d. deployment actions from the fixed task mixture induced by the sampling
protocol. Task-wise coverage and risk are mandatory diagnostics.

No H4 acceptance or test outcome may be reused from H3.
