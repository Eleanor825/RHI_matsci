# H9 Protocol: Train-Only Surrogate Tool Trajectories

Date locked: 2026-08-21

## Motivation

Sequential benchmark v2 generates screening and verification observations by
adding controlled noise around each action's hidden benchmark measurement.
Although the hidden value is never exposed in the visible payload, this is a
semi-synthetic observation mechanism and can overstate the realism of tool
quality. H9 removes this dependency.

## Intervention

For each task and tool stage, fit independent scientific surrogate replicas
using training-split actions and training-split task-native margins only. At
feedback, acceptance, and test time, visible tool outputs are predictions from
those frozen surrogate replicas. Test labels, test utilities, and test hidden
margins must not be used to create any visible observation.

- screening tool: lower-capacity character n-gram ridge surrogate;
- verification tool: higher-capacity character n-gram ridge surrogate;
- external replicas: independently bootstrapped surrogate fits;
- internal replicas: unchanged gain-head replicas conditioned on a fixed tool
  view;
- gain target, normalization, acquisition cost, calibration split, acceptance
  split, risk bounds, coverage grid, and held-out regimes: unchanged from H8.

The exported trajectory provenance must say `train_only_surrogate_prediction`
and must record the train IDs/hash used by the tool. It must not say live
MatBot; this is a real fitted offline tool trajectory, not online deployment.

## Leakage falsification test

After fitting the tool environment, mutate all non-training hidden margins and
labels while keeping visible action fields fixed. Every feedback/acceptance/test
visible tool observation must remain byte-identical. Failure of this test
invalidates H9.

## Primary methods and criteria

Primary method: `task_sparse_continuous_sd_worth` with binary exact selective
risk at `alpha=0.20` and normalized tool cost `0.002`.

H9 succeeds if at least four of five seeds have:

1. nonzero test coverage;
2. empirical binary selective risk at most 0.20;
3. positive population net scientific gain;
4. positive selected mean gain;
5. a passing non-training-outcome invariance audit.

The source-decomposed method is compared with the matched
`task_sparse_continuous_mean` no-source ablation. Superiority is reported as
paired effect sizes and uncertainty, not declared from seed counts alone.
