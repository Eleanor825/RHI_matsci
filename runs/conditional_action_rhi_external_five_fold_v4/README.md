# Conditional Action-Worthiness RHI: external five-fold validation

This is an action-level validation using the common folds available in H17
mixed-log-KVRH pairwise preference, H15 unique-log-KVRH screening, and H14
extreme-phonon discovery. It is not a live or complete MatBot trajectory
benchmark. All methods use task-balanced calibration and the same 10% action
budget.

| Method | Pooled selective risk | Pooled coverage | Pooled ECE | Mutation acceptance |
|---|---:|---:|---:|---:|
| Conditional Action-Worthiness RHI | 0.5417 | 0.9519 | 0.0994 | 1.00 |
| Static-full signals | 0.4457 | 0.7965 | 0.0733 | n/a |

The task-specific mutation was reproducibly proposed and accepted in all five
folds. It added pairwise verification signals: `agent_prior_margin`,
`agent_prior_uncertainty`, `evidence_support`, and `perturbation_stability`.
The RHI mutation is therefore executable, but it is not yet a performance win:
static-full is stronger on pooled risk and calibration in this run.
