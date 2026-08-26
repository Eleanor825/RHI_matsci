# H22 Extreme-Gap Confirmatory Protocol

Frozen on August 23, 2026 after inspecting fold 0 development and before
building or inspecting folds 1--4.

## Task and split

- Official Matbench `expt_gap`, 4,604 compositions.
- Scientific action: advance a composition for extreme high-band-gap
  validation.
- Utility: train-empirical percentile of experimental band gap.
- Extreme outcome: top decile under the train-only gap threshold.
- Outer split unit: chemical system; folds are group-disjoint.
- Fold 0 is development only. Folds 1--4 are confirmatory.

## Numerical harness

The harness combines coarse-composition and full-composition source states,
internal replica disagreement, external source disagreement/OOD, cost, and a
costed high-fidelity measurement. It estimates action gain and expected value
of evidence. It emits numeric estimates only and no route label.

The frozen primary is `source_voi_expected` under verify-before-execute,
`alpha=0.10`, one-sided `delta=0.05`, evidence cost `0.020`, and a 10% action
budget.

## Primary gate

Across folds 1--4, the frozen primary must:

1. certify on acceptance in at least three folds;
2. have positive pooled deployed test population gain;
3. have pooled test selective risk at most `0.10`;
4. beat static `external_variance` in pooled deployed test gain;
5. report the matched `no_source_voi_expected` comparison without replacing
   the frozen primary.

## Fixed-sequence recursive evolution

A secondary self-evolution analysis uses only each fold's acceptance set. The
preordered mutations are:

1. `no_source_voi_expected` -- learned VoI without source-state features;
2. `source_voi_expected` -- add internal/external source-state features;
3. `source_voi_probability` -- mutate the acquisition head.

Candidates are tested in this fixed sequence with a one-sided 95% binomial risk
certificate at `alpha=0.10`. Testing stops at the first failure. Fixed-sequence
testing controls family-wise type-I error at `delta=0.05` without Bonferroni
splitting. Within the certified prefix, a candidate replaces the incumbent only
when acceptance population gain improves by more than `1e-6`. The resulting
contract is then evaluated once on test.

The evolved harness must deploy in at least three folds, have positive pooled
test gain, pooled empirical risk at most `0.10`, and beat static variance.

## Frozen hashes

- `run_external_h22_extreme_gap.py`: `b6605d18cbb8662fcd00f699faacc49f766624024b7fd5ce11f0175ecbcf3d78`
- `external_h22_extreme_gap.py`: `42cac0b0698eef559bcb0b82347451b42884fab59a28c90aeeb89f144cfba079`
- `external_h22_extreme_gap_tools.py`: `b1b21fc22ec6d547ec198b46b8ca77ec7bc4feb813bd95b3bd43f544943fa538`
- shared `run_external_h15_unique_voe.py`: `6e4b10c2dcb26419da934512aa310303f91ff02f793a8946ecf2f31cdd478c71`
