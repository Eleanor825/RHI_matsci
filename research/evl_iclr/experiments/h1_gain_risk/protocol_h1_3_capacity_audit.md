# H1.3 Protocol — Nonlinear Capacity Audit

Status: LOCKED BEFORE EXECUTION; EXPLORATORY  
Locked date: 2026-08-21

## Question

Did H1.1/H1.2 fail because the visible signals are insufficient, or because the linear heads cannot represent nonlinear task/feature interactions?

## Models

Using the unchanged H1.1 target, feature contract, grouped four-way splits, action budget `B=0.10`, coverage grid, and exact risk controller, compare:

- ExtraTrees classifier on budget-conditioned worthiness;
- histogram gradient boosting classifier on budget-conditioned worthiness;
- ExtraTrees regression on continuous budget-conditioned gain, followed by feedback-only Platt calibration of positive-gain probability.

No oracle features may be restored. No model hyperparameters are selected from test results.

## Seeds and Interpretation

Run seeds `{1, 7, 13, 21, 42}`. These seeds have already been inspected in prior experiments, so this is a capacity audit rather than a fresh confirmatory test.

If nonlinear models achieve nonzero exact risk-controlled coverage at `alpha=0.30`, the earlier “signal insufficiency” conclusion is revised to “linear-capacity insufficiency.” If they remain at zero coverage, richer measured internal/tool signals are required.
