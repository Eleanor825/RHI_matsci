# H5 Fresh Confirmatory Protocol

Frozen on August 23, 2026, before inspecting any outcome in the new acceptance
or test partitions.

## Fresh regime

- Source trajectories: `sequential_tool_trajectories_h5_fresh_24000.jsonl`.
- Source SHA-256: `75c5036cded8ab5ba7c7dc5b0bc338a395e1d13e3451e99ded74e915d85d8faa`.
- Exactly 8,000 actions per task and 24,000 actions total.
- Split counts: train 14,630; feedback 3,601; acceptance 2,447; test 3,322.
- Every action is excluded against the prior 6,000-trajectory development set
  using a scientific fingerprint of benchmark, visible context, and candidate
  action. Manifest fingerprint overlap is zero.
- Test trajectories are physically separated from the acceptance runner.

## Frozen numerical harness

For every action and evidence state, the harness emits only numerical values:
`p_success`, conditional utility, expected strict net gain, calibrated
`p(gain>0)`, a conformal lower gain bound, internal variance, external variance,
model-evidence interaction variance, aleatoric variance, and predictive
variance. It emits no route.

Strict trajectory gain is

`S*U - 0.15*(action_cost + acquired_evidence_cost)
     - 0.25*(1-S)*failure_harm`,

and abstention after evidence receives `-0.15*acquired_evidence_cost`.

The factorized gain model is frozen at three bootstrap replicas, 80 trees per
replica, and task-specific heads. Feedback is group-stratified into disjoint
gain-calibration, evidence-calibration, and policy-diagnostic subsets. Action
worthiness uses Platt calibration and gain uses a one-sided normalized split
conformal lower bound at alpha 0.10.

## Frozen downstream allocator

The primary allocator is `shadow_price_successive_halving`:

1. Compute the execution shadow price as the current tenth-percentile budget
   cutoff, i.e. the score of the `ceil(0.10*N)`-th action.
2. Rank screening acquisitions by conservative counterfactual evidence value
   relative to that shadow price, and acquire screening for exactly 20% of the
   batch.
3. Recompute the shadow price, rank verification acquisitions the same way, and
   acquire verification for exactly 10% of the batch.
4. Execute at most 10% of actions by final positive conformal lower gain.

The conservative evidence-value score is the crossed model-by-evidence mean
minus one predictive standard deviation. Evidence costs are charged identically
to every method that consumes the same evidence.

## Confirmatory baselines

- base-only factorized lower gain;
- fixed screening;
- fixed all evidence;
- static reliability;
- shadow-price adaptive mean;
- same-evidence direct LLM judges at base, screening, and full evidence.

## Acceptance gate

The test partition remains locked unless the primary method on fresh acceptance:

1. selects at least one action;
2. has one-sided 95% Clopper--Pearson selective-risk upper bound at most 0.10;
3. has positive total strict trajectory gain;
4. exceeds static reliability and base-only factorized total strict gain;
5. after LLM caches are available, exceeds the strongest same-evidence direct
   LLM judge in total strict trajectory gain without higher empirical risk.

The Clopper--Pearson result is an independent held-out batch certificate, not an
assumption-free action-wise theorem, because batch ranking couples actions.
