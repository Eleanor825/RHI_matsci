# H2.1 Closed-Model × Retrieval-Tool Pilot

Complete matrix: `True`; actions: `30`; cells: `270`.

## Probability quality

| Score | Brier | ECE | AURC | Accuracy |
|---|---:|---:|---:|---:|
| base_ensemble | 0.2033 | 0.2261 | 0.8643 | 0.7333 |
| all_tool_views_ensemble | 0.1654 | 0.1822 | 0.8654 | 0.8333 |

## Source components

| Scope | N | Internal var | External var | Corr internal/error | Corr external/error | Analog shift | Task-stat shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 30 | 0.00253 | 0.01947 | 0.0926 | -0.0376 | -0.1536 | -0.1704 |
| matbench_pairwise | 10 | 0.00463 | 0.05236 | -0.1976 | 0.0061 | -0.4519 | -0.4476 |
| discover_unique | 10 | 0.00243 | 0.00556 | 0.7546 | 0.9580 | -0.0297 | -0.1085 |
| extreme_properties | 10 | 0.00054 | 0.00049 | 0.6553 | 0.1143 | 0.0208 | 0.0451 |
