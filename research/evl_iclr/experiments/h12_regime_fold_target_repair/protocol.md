# H12 Protocol: Stable Opportunity Cost and True Regime Folds

Date locked: 2026-08-22

## Motivation

H10/H11 expose two evaluation defects that must be repaired before further
method claims.

1. The budget reservation is currently the 90th percentile of
   outcome-conditioned base gain. Because the training worthy rate is near
   10%, a small split fluctuation changes the reservation discontinuously. In
   seed 13, `discover_unique` moves from approximately 0.825 to 0 even though
   the underlying utility distribution is unchanged.
2. The five nominal split seeds reuse exactly the same held-out regimes for
   Matbench pairwise and unique discovery. They measure source-partition noise,
   not five distinct scientific-regime generalization tests.

## Target repair

Fit the train-only utility normalizer as before. For each training candidate,
define its potential successful slot value as

```text
candidate_slot_value = normalized_scientific_utility
                       - lambda_cost * normalized_action_cost
```

The task reservation is the `(1 - budget_fraction)` quantile of this value,
clipped below by the zero outside option. This quantity uses historical
training outcomes to define the evaluation target, but it does not multiply by
the binary success label and therefore cannot jump when training success
prevalence crosses the action budget.

Strict net gain remains

```text
G = success * normalized_scientific_utility
    - lambda_cost * normalized_action_cost
    - lambda_harm * (1 - success) * failure_harm
    - reservation_value.
```

No test utility, label, or outcome is used in target fitting.

## Held-out regime folds

Construct five deterministic group folds independently within each task.
Groups are assigned once by size-balanced deterministic bin packing, with each
scientific group appearing in exactly one test fold. Within the non-test groups,
records are deterministically partitioned into train, feedback, and acceptance.
Thus the five reported runs represent distinct held-out scientific regimes,
not repeated random splits of the same test regimes.

The output must record every test group and verify:

- pairwise test groups differ across folds;
- unique-discovery test crystal systems differ across folds;
- extreme-property target IDs differ across folds;
- no group appears in test in more than one fold;
- every benchmark group appears in exactly one test fold.

## Fixed method

Use the H10 real-tool environment, complete 3 x 3 internal/external replica
matrix, task-specific direct tool-margin continuous calibration, feedback-fitted
task-sparse acquisition, and the H11.1 frozen-contract activation rule. The
harness outputs numerical gain-distribution quantities only; routing remains a
downstream budget policy.

Primary method:
`task_sparse_tool_margin_continuous_sd_worth`.

Fixed ablation:
`task_sparse_tool_margin_continuous_mean`.

Static base, evidence heuristic, and cached direct-judge baselines are reported
under the identical fold records whenever their observation contract permits.

## Metrics and gate

Primary: held-out population net scientific gain after tool cost at exact
binary selective-risk target `alpha = 0.20`.

Also report coverage, selected mean gain, binary selective risk, bounded
shortfall, Brier score, ECE, acquisition fraction, per-task coverage, worst
regime risk, and source/no-source paired difference.

The pilot is fold 0. Proceed to all five folds only if the corruption audit
passes and the primary has nonzero coverage, risk at most 0.20, and positive
net gain. The confirmatory target is positive gain and nonzero coverage in at
least four folds, risk at most 0.20 whenever active, selected actions from at
least two of three tasks across folds, and a positive mean paired difference
against the no-source ablation.

H12 is a benchmark-validity repair. A failure is retained as evidence and does
not justify returning to the invalid repeated-regime seed protocol.
