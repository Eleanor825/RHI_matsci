# Combined Action + Trajectory Feedback RHI

This experiment uses complete prompt-generated trajectories. Each action contributes action-level feedback; the full route contributes outcome feedback. One aggregate feedback object produces one candidate harness per RHI round. Individual actions do not independently mutate the harness.

- Trajectories: 900 (300 per task)
- Splits: {'train': 538, 'feedback': 124, 'acceptance': 88, 'test': 150}
- Iterations: 3
- Test outcomes are used only after harness selection.

## Test checkpoints

| Round | Harness | Accepted | Coverage (top 10%) | Risk | Success | Net utility / trajectory | Mean steps |
|---:|---|:---:|---:|---:|---:|---:|---:|
| 0 | H0_rhi_matsci | True | 0.100 | 0.867 | 0.133 | -0.0011 | 2.00 |
| 1 | H1_trajectory_feedback | True | 0.100 | 0.933 | 0.067 | -0.0043 | 2.00 |
| 2 | H1_trajectory_feedback | False | 0.100 | 0.933 | 0.067 | -0.0043 | 2.00 |
| 3 | H1_trajectory_feedback | False | 0.100 | 0.933 | 0.067 | -0.0043 | 2.00 |

## Interpretation

- Action-level feedback identifies local calibration, evidence, and uncertainty failures.
- Trajectory-level feedback identifies wrong final commitments, failed recovery, unnecessary verification, and missed opportunities.
- The proposer mutates the declarative harness once per round; acceptance compares the candidate against its predecessor on a held-out acceptance shard.
- The current prompt backend is deterministic and offline, not an LLM-generated runtime trace; this is a reproducible trajectory benchmark rather than an online MatBot result.
- Runtime-route metrics are also stored in JSON; the table uses a fixed top-10% budget so every checkpoint is compared at equal action budget.
