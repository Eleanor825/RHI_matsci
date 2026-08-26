# H5.1 Protocol: Task-Conditional Acquisition and Risk Certificates

Date locked: 2026-08-21

## Motivation

The three tasks have different positive rates, gain scales, and held-out group
sizes. A single global acquisition threshold and a single global risk
threshold are not invariant to task-mixture shift. H5.1 makes both decisions
task conditional while preserving a global risk statement.

## Method

1. On feedback data, select one base-score acquisition fraction per task from
   `{0.05, 0.10, 0.20, 0.30, 0.50, 1.00}` using total net gain at tool cost
   `0.002`.
2. Freeze the three acquisition thresholds.
3. On acceptance data, select one execution threshold per task using either
   exact binary risk or bounded shortfall risk.
4. Use confidence level `delta / 3` for each task. If every task-conditional
   selected set has risk at most `alpha`, their union has mixture risk at most
   `alpha`; Bonferroni controls simultaneous failure probability by `delta`.

No task-mixture weight from the test split is used.

## Confirmatory success criteria

Across at least four of five seeds, task-conditional sparse continuous
SD-Worth must:

1. obtain nonzero coverage under both binary and shortfall certificates at
   `alpha <= 0.20`;
2. have positive selected mean gain and positive population net gain at cost
   `0.002`;
3. satisfy empirical test mixture risk at most `alpha`;
4. improve worst-task risk over the global-threshold counterpart;
5. match or exceed task-conditional continuous mean on risk-controlled net
   gain in at least three seeds, with aggregate significance evaluated after
   all five seeds.

