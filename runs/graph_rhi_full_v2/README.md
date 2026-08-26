# Graph-RHI on complete materials action trajectories

Records: `20987`; trajectories: `45`.
Split sizes: `{'train': 13148, 'feedback': 2648, 'acceptance': 2622, 'test': 2569}`.
All splits are trajectory-disjoint; the test split is untouched until final evaluation.

| Harness | Top-10% risk ↓ | Hit rate ↑ | Mean utility ↑ | Coverage | Accepted |
|---|---:|---:|---:|---:|:---:|
| H0 fixed | 0.2996 | 0.7004 | 0.1387 | 0.1000 | n/a |
| H1 evidence/source | 0.2957 | 0.7043 | 0.1471 | 0.1000 | False |
| H2 OOD/stability (active) | 0.2879 | 0.7121 | 0.1541 | 0.1000 | True |
| H3 component merge | 0.2879 | 0.7121 | 0.1541 | 0.1000 | False |

Accepted branches: `['H2_ood_stability']`.
True merge accepted: `False`.

H3 is a true merge only when at least two complete parent harnesses pass acceptance and their modified components do not conflict.
