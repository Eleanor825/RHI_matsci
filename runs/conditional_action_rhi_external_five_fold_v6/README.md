# Conditional Action-Worthiness RHI: external five-fold validation

Action-level validation on H17 mixed-log-KVRH pairwise preference, H15 unique-material screening, and H14 extreme-phonon discovery. This is not a live multi-step MatBot trajectory benchmark. All methods use task-balanced calibration and a strict 10% top-k action budget.

| Method | Top-10% risk | Top-10% coverage | Top-10% hit rate | Threshold risk | Mutation acceptance |
|---|---:|---:|---:|---:|---:|
| Conditional RHI | 0.0034 | 0.1000 | 0.9966 | 0.5417 | 1.00 |
| Static-full signals | 0.0034 | 0.1000 | 0.9966 | 0.4457 | n/a |

The conditional RHI candidate was accepted in 100% of folds. The accepted mutation adds pairwise-verification signals: `agent_prior_margin`, `agent_prior_uncertainty`, `evidence_support`, and `perturbation_stability`.

The primary comparison is strict top-10% budget performance. Threshold-based risk is reported separately as a calibration diagnostic.
