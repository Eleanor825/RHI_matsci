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
```

For the confirmatory five-seed run:

```bash
PYTHONPATH=src python3 -m harness_matsci.conditional_action_rhi \
  --actions benchmarks/materials_action_trajectories_v1/actions.jsonl.gz \
  --out-dir runs/conditional_action_rhi_v6_seed_suite \
  --iterations 3 --epochs 60 --seeds 1,7,13,21,42
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

## External action-level validation

To test the signal contract with non-constant scientific features, we also ran
five common folds from the repository's external benchmarks: H17 mixed-log-KVRH
pairwise preference, H15 unique-material screening, and H14 extreme-phonon
discovery. These records are action-level scientific decisions with train,
feedback, acceptance, and test partitions; they are not complete multi-step
MatBot trajectories.

The task-balanced five-fold run is stored in
`runs/conditional_action_rhi_external_five_fold_v6/summary.json`. The active
mutation was reproducibly proposed and accepted in all five folds:

- pairwise verification added `agent_prior_margin`, `agent_prior_uncertainty`,
  `evidence_support`, and `perturbation_stability` as task-specific signals;
- threshold-based pooled selective risk was `0.5417`, coverage `0.9519`, and
  ECE `0.0994`;
- under the strict top-10% budget, both methods had risk `0.0034`, coverage
  `0.1000`, and hit rate `0.9966`;
- mutation acceptance rate was `1.0`.

The strict top-10% result is identical between RHI and static-full, so the
accepted mutation changes the score calibration but not the selected ranking
on this benchmark. The threshold-based risk remains high, and this result
demonstrates executable, repeatable signal evolution rather than a performance
win. The test is an action-level validation of signal evolution, while the
trajectory benchmark is used to audit the no-single-trajectory mutation rule.
