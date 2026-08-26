# H3 Strict Budgeted Gain Confirmatory Protocol

Frozen on August 23, 2026, before generating or inspecting the expanded
confirmatory acceptance and test outcomes.

## Scientific target

For action `i`, the realized net scientific gain is

`G_i = S_i U_i - 0.15 C_i - 0.25 (1-S_i) H_i`.

`S_i` is terminal success, `U_i` is train-distribution normalized scientific
utility, `C_i` is normalized action cost, and `H_i` is irreversible failure
harm. Abstention has gain zero. A train-set top-decile reservation or shadow
price is not subtracted from the realized outcome. Resource scarcity is
represented only by the policy constraint below.

## Harness output

The harness emits numerical quantities only:

- calibrated probability of positive realized gain;
- posterior expected realized gain;
- success probability and conditional utility mean;
- internal, external, and interaction variance from the crossed model/tool
  replica grid;
- aleatoric and total gain variance;
- a one-sided split-conformal lower gain bound.

The harness does not emit a route. A downstream allocator may execute the
selected actions or abstain.

## Primary policy

The primary frozen policy ranks all candidate actions jointly by the
factorized one-sided conformal lower gain bound. It executes at most
`ceil(0.10 N)` actions globally and excludes every action whose lower bound is
not strictly positive. The 10% resource level is a cap, not a quota, and no
per-task action is forced.

## Data regime

- Export 2,000 deterministic records per task for 6,000 trajectories total.
- Use the same three scientific tasks and grouped four-way split protocol.
- Rebuild every training observation with five-fold group-excluded cross-fit
  tool environments.
- Exclude all 3,000 record IDs in
  `sequential_tool_trajectories_v2_3000.jsonl` from confirmatory acceptance and
  test evaluation.
- Expanded train and feedback records may be used for fitting and calibration.
- Expanded acceptance outcomes are used once to accept or reject the frozen
  policy. Expanded test outcomes are inspected only after the acceptance
  decision.

## Models and prompt

Internal replicas are `gpt-5.6-sol`, `gpt-5.6-luna`, and `gpt-5.6-terra` with
prompt version `strict-realized-gain-v1`. Every model estimates strict realized
gain without receiving a reservation value or a route-selection request.

## Baselines

- calibrated direct LLM judge using mean `p_positive_gain`;
- untrained LLM component plug-in gain using `p_success` and conditional
  utility readouts;
- static reliability gate using the frozen uncertainty feature set;
- factorized posterior mean gain without the conformal lower bound.

All methods receive the same global at-most 10% budget. Gain-valued methods
must have positive score; probability-valued methods must exceed 0.5.

## Primary acceptance criterion

On the fresh acceptance set, the primary policy must:

1. select at least one action;
2. have positive selected total and mean realized gain;
3. obtain an exact one-sided 95% Clopper-Pearson selective-risk upper bound no
   greater than `alpha = 0.10`.

The policy and estimator are rejected if any condition fails. The test set
cannot rescue a failed acceptance decision.

## Confirmatory test claims

Conditional on acceptance, the fresh test must show:

- empirical selective risk no greater than 0.10;
- positive selected total and mean net scientific gain;
- higher selected total net gain and lower selective risk than direct LLM
  judge and static reliability;
- one-sided conformal marginal coverage consistent with the 0.90 target;
- non-degenerate internal/external/interaction variance and reported
  correlations with squared gain error.

Paired, task-stratified bootstrap confidence intervals will be reported for
the net-gain differences. This run is confirmatory for the frozen H3 method;
any later method change requires a new held-out round.
