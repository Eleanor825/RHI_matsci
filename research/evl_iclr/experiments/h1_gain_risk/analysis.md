# H1 Analysis

Status: REFUTED on the seed-7 sanity run.

## Sanity Result

- Train/feedback/acceptance/test sizes: `12764/3030/2080/3113`.
- The strict finite-sample risk controller selected zero actions for all three methods at `alpha=0.1, delta=0.05`.
- `binary_worthiness` and `static_reliability` produced identical metrics.
- Across every split, `1[gain > 0]` matched the original success label exactly; mismatch count was zero.

## What This Rules Out

The original gain definition does not create a distinct action-worthiness target on this benchmark. Every successful action remains positive after the cost term, while every failed action is negative. The continuous target can still rank successful actions, but the event `gain > 0` is only success prediction.

The all-prefix Bonferroni exact-binomial controller is also too conservative for the current ranking quality at risk `0.1`; zero coverage is a valid guarantee but not a useful scientific policy.

## Revision

H1.1 will define gain relative to a train-only, task-specific reservation value induced by the available action budget. An action is worthy only if its outcome-conditioned gain exceeds the opportunity cost of consuming a scarce action slot. Risk control will evaluate a predeclared finite coverage grid, preserving a union-adjusted finite-sample guarantee without testing thousands of label-dependent prefixes.
