# H17 Mixed-Regime Pairwise VoE Protocol

Date frozen for fold-0 development: 2026-08-22.

## Purpose

H16 sampled only train-surrogate neighbors. Its visible evidence had essentially
no ability to identify positive value of high-fidelity acquisition. H17 tests the
intended routing problem with a predeclared mixture of scientific evidence
states rather than mutating H16 after observing its outcome.

## Benchmark construction

Materials are split by chemical-system group before pair generation. Models are
fit on the train material split only. Candidate pairs in every split are oriented
by a train-only combined ensemble and assigned, without hidden labels or target
properties, to three regimes:

- easy: large surrogate margin, low ensemble uncertainty, and agreement of
  composition and structure models;
- ambiguous: small surrogate margin with source disagreement or large
  cross-source range;
- OOD: high ensemble uncertainty and non-dominant surrogate margin.

The fixed mixture is 40% easy, 30% ambiguous, and 30% OOD. The hidden target is
the measured Matbench bulk-modulus ordering and normalized utility difference.

## Signals and policy

The base state uses an agent-internal train-only ensemble prior plus two
external train-only source models (composition and structure). A costed
high-fidelity database measurement replay is available after acquisition.
Within-source replica dispersion is internal variance; reliability-weighted
between-source dispersion is external variance.

Feedback evidence-value targets use cross-fitted provisional execution
thresholds. Acquisition scores and the final joint acquisition/base/tool
execution contract are cross-fitted or selected only on feedback. Acceptance
is used exactly once for the exact finite-sample selective-risk certificate.
The certified frozen policy is copied unchanged to test.

Fold 0 is development-only. No untouched fold may be inspected until the method,
cost, coverage grid, alpha, delta, and baselines are frozen.

## Confirmatory freeze after fold-0 development

The primary method is `source_voi_expected`. Its counterfactual VoE posterior is
a five-replica bootstrap ensemble of standardized Ridge regressors (alpha 10.0).
Cross-fitted feedback acquisition scores use three replicas. The auxiliary
positive-VoE classifier uses standardized L2 logistic regression (C 0.10), but
is not the primary score. Each replica uses an 85% bootstrap sample.

The primary comparison is against the matched internal-only/no-source-uncertainty
expected-VoE ablation and the static external-variance acquisition rule. The
coverage grid, tool cost 0.020, empirical feedback risk cap 0.20, acceptance
alpha 0.20, and acceptance delta 0.05 are frozen. Fold 0 is excluded from the
confirmatory aggregate. Folds 1-4 are untouched confirmatory folds and will be
run without method changes. The primary claim requires positive aggregate test
net gain, acceptance certification, and improvement over both primary baselines
across the confirmatory folds.
