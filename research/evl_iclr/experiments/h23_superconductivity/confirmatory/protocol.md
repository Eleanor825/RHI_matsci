# H23 Superconductivity Confirmatory Protocol

Frozen on August 23, 2026 after inspecting fold 0 development and before
building or inspecting folds 1--4.

## Task and split

- Official Matminer `superconductivity2018`, 16,414 source rows.
- Eight rows with invalid composition syntax are removed by a deterministic
  parser rule, leaving 16,406 records.
- Scientific action: advance a composition for extreme high-critical-
  temperature validation.
- Utility: train-empirical percentile of experimental critical temperature.
- Extreme outcome: top decile under the train-only critical-temperature
  threshold.
- Outer split unit: chemical system; folds are group-disjoint.
- Fold 0 is development only. Folds 1--4 are confirmatory.

## Numerical harness

The harness combines coarse-composition and full-composition source states,
internal replica disagreement, external source disagreement/OOD, cost, and a
costed high-fidelity measurement. It estimates action gain and expected value
of evidence. It emits numeric estimates only and no route label.

The frozen primary is `source_voi_expected` under verify-before-execute,
rank-budget policy selection, `alpha=0.10`, one-sided `delta=0.05`, evidence
cost `0.020`, 10% acquisition budget, and 5% post-verification execution
budget.

## Primary gate

Across folds 1--4, the frozen primary must:

1. certify on acceptance in at least three folds;
2. have positive pooled deployed test population gain;
3. have pooled test selective risk at most `0.10`;
4. beat static `external_variance` in pooled deployed test gain;
5. beat matched `no_source_voi_expected` in pooled deployed test gain.

## Fixed-sequence recursive evolution

A secondary self-evolution analysis uses only each fold's acceptance set. The
preordered mutations are:

1. `no_source_voi_expected` -- learned VoI without source-state features;
2. `source_voi_expected` -- add internal/external source-state features;
3. `source_voi_probability` -- mutate the acquisition head.

Candidates are tested in this fixed sequence with a one-sided 95% binomial
risk certificate at `alpha=0.10`. Testing stops at the first failure.
Fixed-sequence testing controls family-wise type-I error at `delta=0.05`
without Bonferroni splitting. Within the certified prefix, a candidate
replaces the incumbent only when acceptance population gain improves by more
than `1e-6`. The resulting contract is evaluated once on test.

The evolved harness must deploy in at least three folds, have positive pooled
test gain, pooled empirical risk at most `0.10`, beat static variance, and beat
the no-source comparator.

## Frozen hashes

- source archive: `03976fc80f39b3d9c7bf2c12f4d045513cdad17a3a2e559f6290fc7245154418`
- `run_external_h23_superconductivity.py`: `c3201c1a59d6cb380adb5c78ae6f2fe2595272a17104c6eb14e90c1687cf3048`
- `external_h23_superconductivity.py`: `1537416a9b867f31550f6cc75bfc3dba4581974ae30e056c1afeb0d9cb68e1a0`
- `external_h23_superconductivity_tools.py`: `dcede8f00ec3ddb628bb77ca78b3118c03516a34766c7d08a887596962472b6a`
- shared `run_external_h15_unique_voe.py`: `6e4b10c2dcb26419da934512aa310303f91ff02f793a8946ecf2f31cdd478c71`
