# H11.1 Frozen-Contract Activation Pilot

The feedback-fitted acquisition contract is frozen before acceptance. The
acceptance split evaluates one nontrivial contract and the zero-cost no-op.

| variant | acceptance coverage | acceptance risk UCB | acceptance net gain | activated | test coverage | test risk | test net gain |
|---|---:|---:|---:|---|---:|---:|---:|
| source decomposed | 0.0202 | 0.1971 | +0.000871 | yes | 0.0151 | 0.0851 | +0.000454 |
| no source variance | 0.0202 | 0.1971 | +0.000871 | yes | 0.0151 | 0.0851 | +0.000454 |

## Interpretation

- The gate preserves the positive H10 seed-7 result instead of losing it to
  acquisition-policy multiplicity correction.
- The binary selective-risk statement is exact over the frozen coverage grid.
- The activation gain check is empirical, not a finite-sample lower-confidence
  guarantee on positive gain; a later protocol must certify both risk and gain.
- Source-decomposed and no-source variants are identical, so this pilot does
  not support an internal/external variance advantage.
- Selected test actions still come only from Matbench pairwise; multi-task
  scientific utility remains unproven.
