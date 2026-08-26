# H21 Unique-Material Alpha-0.10 Confirmatory Protocol

Frozen on August 23, 2026 before building or inspecting folds 1--4.

## Objective

Test whether the route-free numerical uncertainty harness transfers to a
group-disjoint unique-material discovery benchmark under a one-sided 95%
selective-risk requirement of `alpha=0.10`.

## Data regime

- Source: Matbench `log_kvrh` records.
- Split unit: chemical-system group, not individual rows.
- Fold 0: development only and excluded from confirmatory claims.
- Folds 1--4: untouched confirmatory folds.
- Utility: equal-weight train-percentile bulk-modulus performance and
  train-reference composition/structure novelty.

## Frozen primary method

`source_voi_expected`: estimate numerical action gain before and after a
high-fidelity evidence call, estimate expected value of evidence from internal
and external uncertainty state, and derive execution from a fixed numerical
contract. The estimator emits no route label.

`no_source_voi_expected`, probability variants, and `external_variance` are
reported as baselines. They cannot replace the primary method after test
inspection.

## Risk and utility

- Strict scientific gain uses the train-only normalized utility and charges
  the normalized evidence cost `0.020`.
- Feedback chooses thresholds with empirical risk at most `0.10`.
- Acceptance deployment requires a one-sided 95% binomial risk upper bound at
  most `0.10`.
- Test is evaluated only for policies certified on acceptance.

## Confirmatory gate

Across folds 1--4, the frozen primary must:

1. be acceptance-certified in at least three folds;
2. have positive aggregate deployed test population gain;
3. have pooled empirical test risk at most `0.10` among deployed folds;
4. beat `external_variance` in aggregate deployed test gain;
5. report its comparison with `no_source_voi_expected` without replacing the
   primary if the no-source ablation is stronger.

## Frozen implementation hashes

- `run_external_h15_unique_voe.py`:
  `8187e6d1f8600b388c15f4a2e11768275cb5be3999028c79b74c709c9ddef352`
- `external_h15_unique.py`:
  `d19cc66654bbffea54dca575db37dbeaf12c37d8e5a4f80cf261d393c3121823`
- `external_h15_unique_tools.py`:
  `e0291d1c2648d928918bf62a488d3b2d5e2cbe8f1de15a7486ae58454126f690`
