# Paired GPT-5.5 Judge vs Task-Conditional RHI

## Protocol

- Source records: `500` records from the existing real GPT-5.5 direct-judge subset.
- Split: trajectory-safe RHI split using record-level trajectories for this subset; test contains `74` records.
- Fixed action budget: top `10%` of test actions (8 actions).
- The GPT-5.5 values are real cached judge outputs; this run made no new API calls.
- RHI uses only pre-action numeric/context features and does not read judge scores.

## Paired Result

| Method | Risk@10% ↓ | Hit rate ↑ | Mean utility ↑ | Coverage | Mutation accepted |
|---|---:|---:|---:|---:|:---:|
| Task-conditional RHI | 0.7500 | 0.2500 | 0.007865 | 0.1081 | no |
| GPT-5.5 direct judge | 0.6250 | 0.3750 | 0.008380 | 0.1081 | n/a |

## Interpretation

- On this small paired split, GPT-5.5 direct judge selects 3/8 positive actions, while the current RHI selects 2/8.
- RHI did not accept a mutation on this split (`accepted_rounds=[0]`), so this is a fixed-harness comparison rather than evidence of positive self-improvement.
- The result is a valid negative/neutral control: it does not support claiming that the current RHI beats direct LLM judging.
- The external five-fold action-level benchmark remains a separate result and must not be pooled with this subset because the data construction and split protocol differ.

## Reproduction

```bash
PYTHONPATH=src python3 /tmp/run_paired.py
```
