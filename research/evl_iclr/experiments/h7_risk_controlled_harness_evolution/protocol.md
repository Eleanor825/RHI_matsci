# H7 Protocol: Risk-Controlled Recursive Harness Evolution

Date locked: 2026-08-21

## Hypothesis

No single uncertainty harness is uniformly best across scientific regimes.
Recursive harness improvement should therefore evolve the contract on an
independent acceptance set, while preserving a simultaneous selective-risk
guarantee across every candidate harness and coverage threshold considered.

## Candidate sequence

The recursion starts from `base_mean` and considers these mutations in order:

1. `sparse_verification_mean` — add sparse verification and binary mean-gain
   calibration;
2. `sparse_tool_probability` — replace model gain with direct tool success
   probability;
3. `task_sparse_continuous_mean` — add task-specific acquisition and continuous
   gain calibration;
4. `task_sparse_continuous_sd_worth` — add internal/external source variance.

Each candidate is a harness-level contract. The harness is not mutated once
per action.

## Acceptance rule

For five candidate harnesses and eight coverage thresholds, use family-wise
confidence `delta=0.05` by passing `delta/5` to each harness's exact-binomial
grid selector; the selector applies its own correction over the eight
coverages. This makes every considered harness-threshold risk certificate
simultaneously valid under exchangeability.

On the acceptance set, compute population net gain after the candidate's
actual sparse acquisition cost. Accept a mutation only if:

1. it has nonzero certified coverage at `alpha=0.20`;
2. its acceptance population net gain exceeds the incumbent.

Freeze the final harness and threshold before test evaluation.

## Confirmatory criteria

Across at least four of five seeds, evolved H7 must have nonzero test coverage,
test risk at most `0.20`, and positive population net gain. Aggregate mean and
median net gain must exceed the unevolved `base_mean`; comparison with every
fixed candidate is reported without claiming significance from five seeds.

