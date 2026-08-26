# H4.1 Protocol: Task-Specific Gain and Sparse Evidence Acquisition

Date locked: 2026-08-21

## Changes motivated by H4

H4.1 changes only two mechanisms identified before this run:

1. replace the shared gain ensemble with independent task-specific gain heads;
2. call verification only for actions exceeding a base-score acquisition
   threshold learned on the feedback split.

No hidden test outcome may choose the task head, acquisition threshold,
execution threshold, or cost scale.

## Sparse acquisition policy

The harness still outputs only scalar action worthiness. A host-side policy
uses the base score to decide whether to acquire verification evidence. The
candidate acquisition fractions are fixed to
`{0.05, 0.10, 0.20, 0.30, 0.50, 1.00}`. On feedback data, choose the fraction
that maximizes total observed net gain when actions with calibrated
worthiness at least `0.5` are executed. The corresponding base-score
threshold is frozen before the acceptance and test splits.

For an action outside the acquisition set, final action worthiness is zero.
For an acquired action, final action worthiness is the verification-stage
score. Acceptance data are used only to select the exact risk-controlled
execution threshold.

## Cost reporting

Report a full verification-cost sensitivity curve rather than relying on one
arbitrary normalized cost. Predeclared costs are
`{0, 0.0005, 0.001, 0.002, 0.005, 0.010}` per acquired verification. The
break-even cost is also reported directly:

```text
break_even_cost = total selected budget-conditioned gain / acquired actions
```

The primary net-gain comparison uses `0.002`, interpreted as one inexpensive
surrogate verification costing 0.2% of one normalized experimental action.

## Confirmatory success criteria

Across at least four of five seeds:

1. sparse SD-Worth has nonzero exact risk-controlled coverage at
   `alpha <= 0.20`;
2. empirical test selective risk is at most `alpha`;
3. population net gain at verification cost `0.002` is positive;
4. sparse SD-Worth exceeds sparse variance-free mean-gain calibration in
   population net gain;
5. task-specific heads improve selected mean gain over H4's shared head.

