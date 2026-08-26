# H9.1 Protocol: Structured Train-Only Scientific Surrogates

Date locked: 2026-08-21

## Motivation

H9 falsified the generic character-ngram surrogate: it passed the no-test-
outcome audit but had weak worthiness ranking and zero strict coverage. H9.1
tests the mechanistic correction that scientific tools require task-structured
representations rather than generic text similarity.

## Fixed representation

All inputs are visible before action execution.

- Matbench pairwise: sanitized base features, chosen/alternative composition
  element fractions, element-fraction differences, crystal system, and space
  group.
- Unique discovery: sanitized base features, composition element fractions,
  stoichiometric summaries, crystal system, and space group.
- Extreme properties: sanitized base features plus SMILES atom-token counts,
  aromaticity, ring, branch, bond, charge, and heteroatom ratios.

The screening surrogate uses only compact aggregate descriptors. The
verification surrogate additionally uses the full element/token vector.
Independent external replicas are ExtraTrees regressors fitted to bootstrap
weights on training actions only. To prevent extreme-property outliers from
dominating, each task's margin is transformed with `asinh(margin / s_task)`,
where `s_task` is a training-only robust absolute-margin scale; predictions are
mapped back with `sinh`.

## Locked invariants

- no feedback/acceptance/test outcome enters tool fitting or observation;
- explicit hidden-outcome corruption audit must pass;
- gain target, normalization, costs, calibration, risk bounds, split seeds,
  candidate coverage grid, and primary method remain unchanged from H9;
- H9 generic text results remain reported as a negative baseline.

## Pilot and confirmation

Seed 7 is a declared development pilot. If verification worthiness AUC exceeds
H9 on at least two tasks and strict coverage becomes nonzero, the exact locked
H9.1 method is run on seeds 1, 7, 13, 21, and 42. Confirmatory success requires
nonzero coverage, risk at most 0.20, positive selected gain, positive population
net gain, and a passing invariance audit in at least four seeds.
