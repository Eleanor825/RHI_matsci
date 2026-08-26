# H12 True-Regime Fold-0 Pilot

The pilot uses the repaired stable opportunity cost and a genuinely distinct
group-held-out test fold.

## Primary result

| method | acceptance coverage | acceptance empirical risk | acceptance UCB | activated | test net gain |
|---|---:|---:|---:|---|---:|
| source-decomposed real-tool harness | 0.0000 | 0.0000 | 0.0000 | no | 0.000000 |
| no-source real-tool ablation | 0.0000 | 0.0000 | 0.0000 | no | 0.000000 |

The H12 pilot gate fails, so folds 1--4 are not run under this protocol.

## Acceptance-grid diagnosis

| candidate coverage | selected | errors | empirical risk | exact UCB | certified |
|---:|---:|---:|---:|---:|---|
| 0.0055 | 10 | 0 | 0.0000 | 0.3980 | no |
| 0.0105 | 19 | 0 | 0.0000 | 0.2344 | no |
| 0.0205 | 37 | 2 | 0.0541 | 0.2213 | no |
| 0.0504 | 91 | 27 | 0.2967 | 0.4298 | no |

## Interpretation

- The corruption audit passes for all three tasks.
- The repaired reservations are stable and all test regimes are genuinely
  held out.
- The top 1% of acceptance actions is perfectly ranked empirically, but 19
  samples are insufficient for the eight-candidate family-wise exact
  certificate at alpha 0.20.
- The top 2% also narrowly misses certification; its UCB is 0.2213.
- Ranking degrades sharply at 5% coverage, so more acceptance data can recover
  only a narrow safe region. It cannot by itself create broad useful coverage.
- Source-decomposed and no-source variants remain decision-identical.

H12 is a negative but more credible result than the legacy repeated-regime
five-seed experiment. The next protocol may increase the independent
acceptance allocation while treating fold 0 as development only; it must not
lower alpha, remove multiplicity correction, or reuse fold 0 as confirmatory
evidence.
