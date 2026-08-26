# H2.2 Protocol — Candidate-Specific Simulator Tool Pilot

Status: LOCKED BEFORE EXECUTION; DEVELOPMENT PILOT  
Locked date: 2026-08-21

## Motivation

The H2.1 task-statistics tool improved Brier score by shrinking probabilities but did not improve action ranking. The replacement tool must be candidate-specific and must not use the target action's hidden outcome.

## Fixed Actions and Agent Replicas

- Reuse the same deterministic 30 held-out actions from H2.1.
- Reuse identical `gpt-5.6-sol`, `gpt-5.6-luna`, and `gpt-5.6-terra` prompts and settings.
- Reuse cached base-evidence and nearest-analog cells; request only the new simulator view.

## Candidate Simulator

For each task, train three classifiers and three gain regressors on the train split only:

- ExtraTrees;
- Random Forest;
- Histogram Gradient Boosting.

Calibrate the mean classifier probability on the feedback split only. The simulator returns candidate-specific positive-gain probability, expected budget-conditioned gain, classifier disagreement, and gain disagreement. The target test action's label, utility, and gain are never used.

## Evidence Matrix

Use exactly three evidence views:

1. base evidence;
2. train-only nearest historical analogs;
3. train-only candidate-specific simulator ensemble.

Every action requires all nine model-by-tool cells.

## Primary Outcome

The primary diagnostic is AURC/ranking, not Brier score. The simulator tool is considered useful only if it improves AURC over both H2.1 base ensemble (`0.8643`) and task-prior tool ensemble (`0.8654`) and improves top-budget net gain without target leakage.

This reuses an inspected development subset and is not a confirmatory result.
