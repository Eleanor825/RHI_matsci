# Frozen hybrid product-gain protocol

Frozen on 2026-08-22 before complete LLM predictions are available.

For each pairwise scientific action, three GPT-5.6 variants provide independent
`p(A better)` samples. Their mean determines the proposed candidate, and each
probability is reoriented to `P(proposal succeeds)`. Separately, the train-only
bootstrap evidence model provides 12 replicas across four scientific evidence
channels. Absolute source predictions estimate the candidate utility-gap
magnitude and are weighted by candidate-excluded train-only source reliability.

Let `X=2P(success)-1` and `M=|Delta U|`. The harness evaluates the explicit
crossed product `G=XM-0.01`. It reports the exact decomposition

`Var(G) = E[M]^2 Var(X) + E[X]^2 Var(M) + Var(X) Var(M)`

as internal, external, and interaction uncertainty. Feedback fits a shared
continuous gain calibrator. Half of acceptance fits a one-sided normalized
split-conformal lower bound; the other half performs one exact binomial test of
the fixed `LCB>0` policy at `alpha=0.10`, `delta=0.05`. Test is evaluated only
after this contract is frozen.

Baselines are raw three-model direct probability, static deterministic utility,
internal-only bootstrap ensemble, and the deterministic crossed evidence model.
Each individual LLM is scored on the candidate it independently selects; the
three-model direct ensemble is scored on the same consensus proposal used by
the hybrid. Component ablations retrain the calibrator with mean only,
internal variance only, internal plus external variance, and the complete
internal/external/interaction decomposition. These ablations are diagnostic;
only the predeclared complete method is eligible for deployment certification.
The primary hybrid claim requires better Brier or log loss than raw direct LLM,
valid conformal coverage, and positive certified test net gain. Failure to pass
the exact acceptance certificate deploys zero test actions.
