# H5 Cost-Fair Adaptive-Evidence Protocol

Frozen design note recorded on August 23, 2026, before any H5 acceptance or
test outcomes are generated.

## Audit motivation

H3 gives the factorized harness screening and verification observations while
the direct-judge baseline sees only the base view, and H3 strict gain does not
charge the two evidence-acquisition costs. H3 remains a diagnostic of whether
crossed evidence can rank actions, but it is not the primary cost-fair baseline
comparison.

## Numerical harness

At evidence state `s`, the harness emits only numerical estimates:

- `p_success(s)`;
- `expected_utility_given_success(s)`;
- `expected_execute_gain(s)`;
- `lower_execute_gain(s)`;
- internal, external, interaction, aleatoric, and total variance;
- a conservative value estimate for each still-unobserved evidence source.

It does not emit `selected_route`. A frozen downstream allocator consumes these
numbers.

## Strict trajectory gain

For a trajectory that acquires evidence set `E` and then executes,

`G_traj = S U - 0.15 (C_execute + sum_{e in E} C_e)
          - 0.25 (1-S) H`.

For a trajectory that acquires evidence and then abstains,

`G_traj = -0.15 sum_{e in E} C_e`.

Screening costs `0.025`; verification costs `0.075`. These costs are identical
for every method that consumes the corresponding observation.

## Cost-fair baselines

1. base-only direct LLM judge, charged no evidence cost;
2. fixed-screening direct LLM judge, shown the screening observation and
   charged screening cost;
3. fixed-all-evidence direct LLM judge, shown the same screening and
   verification bundle as the full harness and charged both costs;
4. static reliability trained on the same visible evidence bundle and charged
   the same costs;
5. factorized mean gain without conservative calibration;
6. adaptive numerical harness with the frozen downstream allocator.

Every method has the same execution budget and hidden outcomes remain absent
from all prompts and pre-action features.

## Adaptive evidence value

For an unobserved source `e`, define conservative evidence value

`VoE_e(s) = E[max(0, L_execute(s union O_e)) | s]
            - max(0, L_execute(s)) - 0.15 C_e`.

The allocator acquires evidence only when its conservative value is positive.
It executes only when the final lower execute gain is positive, otherwise it
abstains. This routing rule is downstream of the numerical harness and is held
fixed during evaluation.

## Data and claims

- Train on group-disjoint controlled tool trajectories.
- Use real model calls at base, screening, and verification stages.
- Use a fresh acceptance set once for risk certification and unlock test only
  after passing.
- Report task-wise selection, evidence usage, cost, risk, and gain.
- Historical LFP replay and live MatBot trajectories remain separate external
  validity tests.

The primary H5 claim requires higher total strict trajectory gain and lower
selective risk than the same-evidence direct judge, not merely the base-only
judge.
