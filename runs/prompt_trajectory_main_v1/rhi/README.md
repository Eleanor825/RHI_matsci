# Trajectory RHI Run

This is a regime-held-out RHI run over the fixed 900-trajectory source actions. The harness is mutated once per RHI iteration, not once per action.

Split sizes: `{'train': 538, 'feedback': 124, 'acceptance': 88, 'test': 150}`
Accepted versions: `['H0_rhi_matsci', 'H1_trajectory_feedback']`

| Round | Harness | Accepted | Test score |
| ---: | --- | --- | ---: |
| 0 | `H0_rhi_matsci` | True | 0.5269 |
| 1 | `H1_trajectory_feedback` | True | 0.5231 |
| 2 | `H1_trajectory_feedback` | False | 0.5231 |
| 3 | `H1_trajectory_feedback` | False | 0.5231 |

## Interpretation

- Each action contributes feedback; actions do not independently mutate the harness.
- One proposal is generated from aggregate feedback at each RHI round.
- The acceptance set decides whether that proposal becomes the active harness.
- This run uses source action records extracted from the prompt trajectories; the next stronger version should aggregate multi-step route outcomes before proposal generation.
