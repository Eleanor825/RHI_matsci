# H5 Round-2 Confirmatory Protocol: Feedback-Calibrated Portfolio KG

Frozen on August 23, 2026, before exporting or inspecting any outcome in the
round-2 acceptance or test partitions.

## Scientific decision problem

The agent faces a shared scientific execution budget. For action `i`, evidence
stage `z`, success `S_i`, normalized conditional utility `U_i`, action cost
`C_i`, acquired evidence cost `E_i(z)`, and failure harm `H_i`, strict realized
scientific gain is

`G_i(z) = S_i U_i - 0.15(C_i + E_i(z)) - 0.25(1-S_i)H_i`.

An unexecuted action contributes only the evidence cost already acquired:
`-0.15 E_i(z)`. The downstream executor may select at most 10% of the batch.
Evidence acquisition is also capped: at most 20% receive screening and at most
10% receive verification.

## Frozen numerical harness

For every action and evidence state, the harness emits numerical quantities
only: success probability, conditional utility, expected strict gain,
calibrated probability of positive gain, one-sided conformal lower gain,
internal variance, external variance, model-evidence interaction variance,
aleatoric variance, and total predictive variance. It emits no route.

The gain estimator has task-specific factorized success and conditional-utility
heads with three bootstrap replicas and 80 ExtraTrees per replica. Gain
worthiness uses task-stage Platt calibration. Gain lower bounds use one-sided
normalized split conformal calibration at alpha 0.10.

## Primary acquisition rule

The primary method is `learned_portfolio_kg_mean` under a shared global budget.
It fixes the mismatch found in round 1 between evidence acquisition and final
execution:

1. Convert every current and counterfactual future gain cell to the same
   sunk-cost-neutral conformal execution score.
2. For each action, compute its exact one-action top-k replacement threshold:
   the leave-one-out shadow price of the best alternative portfolio.
3. Use the crossed model-by-evidence grid to form a structural one-step
   portfolio knowledge-gain estimate, charging the incremental evidence cost.
4. Train a task-transition bootstrap residual head on outcome-linked feedback.
   Its target is the realized change in strict top-k portfolio gain caused by
   acquiring evidence for that action, not a route label.
5. Acquire screening or verification only when predicted incremental portfolio
   gain is positive, subject to the 20%/10% acquisition caps.
6. Execute at most 10% of actions by final positive conformal lower gain.

The task-balanced variant is a prespecified stress test, not the primary shared-
budget claim.

## Development evidence used to freeze the method

- On the prior 6,000-action development benchmark, the primary method improved
  strict trajectory gain over base-only on both old acceptance and old test.
- On the independent round-1 policy-feedback partition, the earlier global
  implementation improved strict trajectory gain from 81.0043 to 84.0533 and
  reduced empirical selective risk from 0.1121 to 0.0862.
- The round-1 acceptance and locked test are not eligible as round-2 evidence.

## Round-2 data separation

- Training and calibration use only the existing round-1 train and feedback
  partitions.
- Round-2 candidate records are exported after this protocol is frozen.
- Every round-2 record must have zero scientific-fingerprint overlap with both
  the prior 6,000-action development benchmark and all 24,000 round-1 records.
- The scientific fingerprint is benchmark plus normalized visible context plus
  normalized candidate action.
- Round-2 test outcomes remain physically separated and are not passed to the
  acceptance runner.

## Confirmatory baselines

- base-only factorized conformal lower gain;
- fixed screening for every action;
- fixed full evidence for every action;
- static reliability;
- round-1 shadow-price successive halving;
- structural decision-consistent KG without the learned feedback head;
- same-evidence direct LLM judges for GPT-5.6 Sol, Luna, and Terra at base,
  screening, and verification evidence states.

## Acceptance gate

The round-2 test remains locked unless the primary method on fresh acceptance:

1. selects at least one action;
2. has one-sided 95% Clopper--Pearson selective-risk upper bound at most 0.10;
3. has positive strict trajectory gain;
4. exceeds base-only, fixed screening, fixed full evidence, static reliability,
   and round-1 successive halving in strict trajectory gain;
5. exceeds the strongest same-evidence direct LLM judge in strict trajectory
   gain without higher empirical selective risk.

The Clopper--Pearson statement is an independent held-out batch certificate.
It is not claimed as an assumption-free action-wise guarantee because top-k
portfolio ranking couples actions.

## Frozen implementation hashes

- `run_h5_cost_fair_adaptive_evidence.py`:
  `8394283fb72bfca3d40115da4d32059dff5a5ea04ed39f0640b0a74fb4495018`
- `adaptive_gain_harness.py`:
  `2874db7fd58b656f9d6314de65b7d5a7297f21dfea29ebcf278d40a6e745be02`
- `portfolio_knowledge_gradient.py`:
  `d7bbe53c7675b503508877a4662b2987fcc2e5538e75b32a14a3961daf147041`
- `adaptive_factorized_model.py`:
  `204bfeda8178183a67e6b8ed87e56de3acf513c22fb83cc602121b326eb01331`
- `adaptive_evidence_dataset.py`:
  `cf48359a8f87a78a81c0a8b9a952067f2ad80e573594e7e41ff074f9c704020f`
