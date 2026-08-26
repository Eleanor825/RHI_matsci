# H18 Dielectric Verify-Before-Execute Confirmatory Protocol

Protocol frozen: 2026-08-22.

## External task

H18 uses the Matbench dielectric dataset, which is distinct from the log-KVRH
pairwise development dataset. Materials are split by chemical-system group
before pair generation. A train-only ensemble and train-only composition and
structure regressors assign candidate pairs to a fixed 40% easy, 30% ambiguous,
and 30% OOD mixture. Pair assignment does not use measured dielectric outcomes.
Each material split generates three pairs per material to provide sufficient
power for the exact risk certificate.

## Frozen method

The base numeric state contains an agent-prior ensemble plus composition and
structure evidence. Within-source replica dispersion is internal variance;
between-source disagreement and agent-prior/external gaps are external
uncertainty. A costed high-fidelity dielectric measurement replay is available
only after acquisition.

The execution contract is verify-before-execute: base execution coverage is
fixed to zero. Acquired actions may execute after the high-fidelity observation.
Feedback VoE targets use cross-fitted provisional execution thresholds. The VoE
posterior is a regularized linear bootstrap ensemble. The primary acquisition
score is `source_voi_probability`, the calibrated probability that acquiring
evidence has positive counterfactual net value.

The matched primary ablation is `no_source_voi_probability`. Static external
variance is the static uncertainty baseline. `source_voi_expected` and
`no_source_voi_expected` are secondary diagnostics.

## Frozen evaluation

- tool cost: 0.020 normalized gain units;
- acquisition/tool coverage grid: 0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0;
- feedback empirical selective-risk cap: 0.20;
- acceptance certificate: exact binomial upper bound with alpha 0.20 and delta 0.05;
- acceptance is used exactly once and the certified policy is copied unchanged
  to test;
- fold 0 is development-only and excluded from confirmatory aggregation;
- folds 1-4 are untouched confirmatory folds.

The confirmatory primary passes only if it has positive mean deployed test gain,
certifies at least three of four folds, exceeds matched no-source probability in
mean deployed gain, and static external variance does not match its certified
activation and gain.
