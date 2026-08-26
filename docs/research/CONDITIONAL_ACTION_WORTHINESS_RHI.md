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
  --out-dir runs/conditional_action_rhi_v6_seed_suite \
  --iterations 3 --epochs 120

For the confirmatory five-seed run:

```bash
PYTHONPATH=src python3 -m harness_matsci.conditional_action_rhi \
  --actions benchmarks/materials_action_trajectories_v1/actions.jsonl.gz \
  --out-dir runs/conditional_action_rhi_v6_seed_suite \
  --iterations 3 --epochs 60 --seeds 1,7,13,21,42
```
```

The benchmark contains 20,987 reconstructed action records from three task
families. The records are historical benchmark-derived proxies, not live
MatBot runtime logs. Trajectory identities are held out before splitting so
that actions from one reconstructed trajectory do not cross train, feedback,
acceptance, and test partitions.

## Current run

The confirmatory run uses five seeds (`1, 7, 13, 21, 42`), a task-stratified
trajectory split, three RHI rounds, a 10% action budget, and target selective
risk 0.10. It compares the conditional RHI against a static-full signal
baseline using the same split for every seed.

| Method | Selective risk | Coverage | ECE | Brier | Accepted mutation rate |
|---|---:|---:|---:|---:|---:|
| Conditional Action-Worthiness RHI | 0.3452 ± 0.0202 | 0.1285 | 0.0759 | 0.1745 | 0.00 |
| Static-full signal baseline | 0.1946 | 0.1285 | 0.0786 | not reported | not applicable |

The RHI loop generated a task-stage candidate (`materials_pairwise_preference`
verification → `evidence_support`) but rejected it on the held-out acceptance
set in all five seeds. This is a valid safety outcome, but it means this run
does not demonstrate a positive self-improvement gain. The conditional H0 is
also weaker than the static-full baseline, so the current benchmark is not
evidence for an ICLR-level performance claim yet.

The high risk and low effective coverage indicate that the historical
benchmark-derived signals and trajectory split are not sufficient to support a
strong runtime risk guarantee. The next required experiment is to add real
task-stage trajectory variation and a policy that learns evidence value, then
retest the same frozen acceptance protocol.
