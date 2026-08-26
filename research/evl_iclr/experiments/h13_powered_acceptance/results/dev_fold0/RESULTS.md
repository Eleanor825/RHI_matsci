# H13 Powered-Acceptance Development Fold

Fold 0 is development-only and is excluded from confirmatory inference.

| method | acceptance selected | acceptance risk | exact UCB | test coverage | test risk | selected gain | population net gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| source-decomposed primary | 36 | 0.0278 | 0.1831 | 0.0051 | 0.0333 | 0.0706 | +0.000228 |
| no-source ablation | 36 | 0.0278 | 0.1831 | 0.0051 | 0.0333 | 0.0690 | +0.000220 |
| base reliability | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| ensemble LCB | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| verbal confidence | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| evidence heuristic | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |

## Gate decision

- Corruption audit: pass for all three tasks.
- Nonzero exact-risk-controlled coverage: pass.
- Empirical test risk at most 0.20: pass.
- Positive selected mean gain: pass.
- Positive population net gain after tool cost: pass.

The H13 feasibility gate passes. The complete contract is frozen before
untouched folds 1--4.

## Limits

- Selected test actions are still Matbench pairwise only.
- The source-decomposed gain advantage over no-source is only `+0.00000817`.
- Fold 0 is adaptive development evidence and cannot be included in the final
  confirmatory statistical comparison.
