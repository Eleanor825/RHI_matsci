# H13 Protocol: Powered Acceptance on Untouched Regime Folds

Date locked: 2026-08-22

## Motivation

H12 fold 0 uses 1,805 acceptance actions. The primary top-1% prefix has zero
errors among 19 actions but an exact family-wise UCB of 0.2344; the top-2%
prefix has 2/37 errors and UCB 0.2213. Both narrowly miss alpha 0.20. The 5%
prefix is a genuine ranking failure and is not targeted by this experiment.

## Fixed change

Retain the H12 stable opportunity-cost target, five deterministic group folds,
real scientific tools, 3 x 3 replica matrix, model classes, acquisition policy,
coverage grid, alpha 0.20, delta 0.05, tool cost, and frozen-contract activation.

Change only the non-test source allocation from:

```text
train=0.60, feedback=0.15, acceptance=0.10
```

to:

```text
train=0.50, feedback=0.15, acceptance=0.20
```

The ratios are normalized within the non-test groups. This approximately
doubles independent acceptance power while retaining a large train partition.

## Development and confirmation

- Fold 0 is development-only because its H12 acceptance outcomes were already
  inspected.
- Run H13 on fold 0 only as a feasibility check.
- If fold 0 has nonzero exact-risk-certified coverage, empirical risk at most
  0.20, positive selected mean gain, positive population net gain after tool
  cost, and a passing corruption audit, freeze the entire contract.
- Only then run untouched folds 1--4 as confirmatory evidence.
- Fold 0 is never pooled into confirmatory significance tests.

## Baselines

Under identical records and risk control, report base reliability, verbal
confidence, evidence heuristic, ensemble LCB, no-source continuous calibration,
and the source-decomposed primary. Closed LLM judges remain pending until a
complete score matrix is available for the same fold records.

## Confirmatory target

Across untouched folds 1--4:

- exact selective risk at most 0.20 whenever active;
- nonzero coverage and positive net gain in at least three folds;
- positive mean paired gain against base reliability and no-source ablation;
- selected actions from at least two scientific tasks across the four folds;
- all hidden-outcome corruption audits pass.

Failure is retained. Alpha, multiplicity correction, fold membership, and tool
cost may not be relaxed after observing confirmatory folds.
