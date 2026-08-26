# Conditional Action-Worthiness RHI

## Method

The method treats each MatBot proposal as one action-level decision unit. A
single host agent produces the action; the harness does not use a multi-agent
ensemble. At decision time it reads only visible context, evidence, task,
trajectory stage, and cost/risk signals. It predicts action worthiness and
selective risk with task- and stage-conditioned features.

The RHI loop aggregates feedback from many actions and trajectories. A single
trajectory cannot change the global contract. A task-stage pattern must have at
least 20 feedback actions and a higher mean absolute calibration error than the
global feedback population before it can propose a signal mutation. Candidate
mutations are refit on training plus feedback data and compared with the
predecessor on an independent acceptance split. Rejected candidates leave the
active harness unchanged.

## Reproduction

```bash
PYTHONPATH=src python3 -m harness_matsci.conditional_action_rhi \
  --actions benchmarks/materials_action_trajectories_v1/actions.jsonl.gz \
  --out-dir runs/conditional_action_rhi_v1 \
  --iterations 3 --epochs 120
```

The benchmark contains 20,987 reconstructed action records from three task
families. The records are historical benchmark-derived proxies, not live
MatBot runtime logs. Trajectory identities are held out before splitting so
that actions from one reconstructed trajectory do not cross train, feedback,
acceptance, and test partitions.

## Current run

The first full run uses seed 1729, a 60/15/10/15 trajectory split, three RHI
rounds, a 10% action budget, and target selective risk 0.10. It produced:

- train: 14,349 actions from 27 trajectories;
- feedback: 2,504 actions from 6 trajectories;
- acceptance: 860 actions from 5 trajectories;
- test: 3,274 actions from 7 trajectories;
- initial active signals: evidence support, verbal confidence, cost, and reversibility;
- the candidate mutation did not pass acceptance, so the active harness remained H0;
- test coverage: 0.0412;
- test selective risk: 0.4296;
- test ECE: 0.0244;
- fixed-budget hit rate: 0.6483.

The high held-out risk relative to the nominal 0.10 target is a distribution
shift warning, not a positive risk-guarantee result. It motivates regime-wise
calibration and more independent runtime trajectories before a paper claim.
