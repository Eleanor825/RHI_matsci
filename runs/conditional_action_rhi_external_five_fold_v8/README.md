# Fixed Harness vs Evolved Harness: Five-Fold Control

## Protocol

- Tasks: H17 mixed-log-KVRH pairwise preference, H15 unique-material screening, and H14 extreme-phonon discovery.
- Folds: common folds `0--4`.
- Budget: fixed top `10%` of test actions.
- Target selective risk: `alpha=0.10`.
- RHI: three feedback/mutation rounds with held-out acceptance.
- `H0 fixed`: same initial three signals (`verbal_confidence`, `cost`, `reversibility`), same fold and training protocol, but mutation disabled.
- `static-full`: all candidate signals, included as an information-rich reference rather than the primary self-evolution control.

## Results

| Method | Top-10% risk ↓ | Top-10% hit rate ↑ | Coverage |
|---|---:|---:|---:|
| H0 fixed, no evolution | 0.0082 | 0.9918 | 0.1000 |
| Task-conditional RHI | 0.0112 | 0.9888 | 0.1000 |
| Static-full reference | 0.0034 | 0.9966 | 0.1000 |

The RHI mutation was accepted in all five folds (`1.00` acceptance rate), but
its held-out top-budget risk is `0.0030` higher than H0 on average. Therefore,
this experiment does **not** show a performance gain from evolution. It shows
that the current acceptance objective can deploy a materially different policy
without improving the final top-budget metric on this action-level benchmark.
The result motivates a stricter paired acceptance rule and trajectory-level
utility evaluation.

Full machine-readable output is in `summary.json`.
