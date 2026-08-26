# H19 Frozen External Validation Protocol

## Status and separation

- Frozen before any H19 benchmark fold is constructed or evaluated.
- Dataset: `matbench_phonons`, target `last phdos peak`.
- H19 is not used to select the estimator, contract, acquisition head, cost, risk level, coverage grid, or success criterion.
- The primary method was selected only on the five consumed H17 log-KVRH evolution folds.
- H18 dielectric results are external evidence but are not used to tune H19.
- All five H19 folds, 0 through 4, are confirmatory.

## Data regime

- Five outer folds formed by chemical-system group, with no chemical-system overlap among train, feedback, acceptance, and test within a fold.
- Remaining non-test groups are partitioned into train, feedback, and acceptance with fixed weights 0.50, 0.15, and 0.20.
- Pair generation uses only train-fitted surrogate predictions to form the fixed easy, ambiguous, and OOD mixture.
- True phonon targets are hidden from pair construction and from all pre-action views.
- Five pairs are generated per material, fixed before outcome inspection to provide certificate power on the smaller phonon dataset.
- Random seed: 1901.

## Numeric uncertainty state

The harness estimates calibrated action worthiness and source-conditioned uncertainty from:

- an agent-prior view;
- a composition-only train-fitted source model;
- a structure-only train-fitted source model;
- a costed high-fidelity measurement replay available only after acquisition.

The learned acquisition score is expected value of evidence. Internal and external uncertainty features are numeric inputs; the estimator does not emit a route label.

## Frozen primary

- Contract: `verify_before_execute`.
- Base execution coverage: exactly 0; an action may execute only after evidence acquisition.
- Acquisition head: `source_voi_expected`.
- Evidence-value target: cross-fitted policy-conditioned counterfactual net gain.
- Model family: regularized linear.
- Tool cost: 0.020 normalized population utility units.
- Tool execution coverage grid: 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0.
- Selective-risk target: alpha = 0.20.
- One-sided certificate failure probability: delta = 0.05.
- A fold deploys only when its independent acceptance split certifies the selected feedback policy.
- A non-certified fold receives deployed test gain zero.

## Frozen baselines

1. `no_source_voi_expected`: identical pipeline with all external-source disagreement features removed.
2. `external_variance`: static acquisition by calibrated external variance under the same policy search and certificate.

Probability and tail heads may be logged as secondary diagnostics but are not confirmatory alternatives and cannot replace the primary after results are observed.

## Confirmatory success criterion

The H19 transfer claim succeeds only if all conditions hold:

1. `source_voi_expected` certifies on at least four of five folds;
2. its mean deployed test population gain is positive;
3. its mean deployed test population gain exceeds `no_source_voi_expected`;
4. its mean deployed test population gain exceeds `external_variance`.

Failure of any condition is reported as a failed H19 confirmatory transfer, without changing the primary method.
