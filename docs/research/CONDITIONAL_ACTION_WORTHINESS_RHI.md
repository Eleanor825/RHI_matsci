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

## v7 task-conditional ranking-head experiment

This follow-up implements a substantive harness mutation rather than only
adding interaction features. After a mutation is accepted, the harness fits
one score head per task (`pairwise`, `unique`, and `extreme`) and uses that
head for ranking actions from the corresponding task; unseen or data-poor
tasks use the pooled fallback head. Threshold calibration remains grouped by
task and is fit only on feedback data. The mutation is accepted only when its
acceptance-split score improves while respecting the predecessor risk guard.

The confirmatory run uses the three reusable external datasets, common folds
`0--4`, three RHI rounds, 80 training epochs, target risk `alpha=0.10`, and a
fixed top-10% action budget. Results are in
`runs/conditional_action_rhi_external_five_fold_v7/summary.json`; per-fold
artifacts and the static baseline are in the same directory.

| Method | Top-10% risk | Top-10% hit rate | Top-10% mean utility | Coverage | Accepted mutation rate |
|---|---:|---:|---:|---:|---:|
| Task-conditional RHI heads | 0.0112 ± 0.0076 | 0.9888 ± 0.0076 | 0.7456 ± 0.0237 | 0.1000 | 1.00 |
| Static-full pooled head | 0.0034 | 0.9966 | not retained | 0.1000 | n/a |

The accepted mutation changes the decision policy materially: mean top-budget
rank displacement is `0.4966` (about half of selected actions are replaced
relative to the predecessor). However, it does not yet improve held-out
scientific utility on these external records: static-full has lower error and
higher hit rate. Thus v7 validates that RHI can learn and deploy a genuinely
task-conditioned ranking policy, but it is not a positive performance claim.
The next decisive test is trajectory-level online evaluation where evidence
collection changes later action states; the current external files are still
action-level historical proxies.

## H0 fixed versus evolved RHI control (v8)

The key self-evolution control uses the same five common folds and the same
three external task datasets as v7. `H0 fixed` trains the initial harness with
only `verbal_confidence`, `cost`, and `reversibility` and disables all RHI
mutations. The evolved condition uses three RHI rounds and the held-out
acceptance protocol. Both select the top 10% of test actions.

| Method | Top-10% risk | Top-10% hit rate | Coverage |
|---|---:|---:|---:|
| H0 fixed, no evolution | 0.0082 | 0.9918 | 0.1000 |
| Task-conditional RHI | 0.0112 | 0.9888 | 0.1000 |
| Static-full reference | 0.0034 | 0.9966 | 0.1000 |

The H0-versus-RHI comparison is the relevant test of the value of evolution.
Although the mutation was accepted in all five folds and changed the ranking
policy, RHI is `0.0030` risk points worse than H0 on average. The current
acceptance score is therefore not yet aligned with held-out top-budget
scientific utility. This is a negative result for the present evolution rule,
not evidence that RHI itself improves performance. The result is stored in
`runs/conditional_action_rhi_external_five_fold_v8/summary.json`.

## Graph-RHI branching and merge pilot

We implemented a graph-structured evolution protocol in
`src/harness_matsci/graph_rhi.py`. `H0` is the fixed initial harness. `H1`
and `H2` are independently trained mutation branches targeting complementary
failure modes: evidence/source reliability and OOD/perturbation stability.
Each branch is evaluated against `H0` on the held-out acceptance split. A true
`H3` merge is allowed only if at least two independent branches pass acceptance;
otherwise the best accepted branch, or `H0`, remains active.

The five-fold result is stored in
`runs/graph_rhi_external_five_fold_v1/summary.json`.

| Node | Top-10% risk | Top-10% hit rate | Mean utility | Acceptance rate |
|---|---:|---:|---:|---:|
| H0 fixed | 0.0082 | 0.9918 | 0.6865 | n/a |
| H1 evidence/source | 0.0112 | 0.9888 | 0.7456 | 1.00 |
| H2 OOD/stability | 0.0146 | 0.9854 | 0.6207 | 0.00 |
| H3 true merge | not activated | not activated | not activated | 0.00 |

This is an implementation and protocol pilot, not evidence that graph merging
improves performance. The current data accepts only H1 in all five folds, so
there is no genuine two-parent merge yet. The result is useful because it
exposes the required next experiment: construct trajectory-level feedback in
which the two failure modes are complementary, then test whether a two-parent
merge improves the Pareto frontier over linear RHI and branch-only selection.

### Clarification: harness graph semantics

In the graph formulation, `H0`, `H1`, `H2`, and `H3` denote complete harness
versions. They are not independent classifiers or collections of signals. A
harness node contains the shared agent interface, action/decision unit,
calibration rule, acceptance rule, budget, signal contract, and routing policy.
H1 and H2 inherit the shared components from H0 while modifying different
components. H3 reuses the compatible shared components and combines only
non-conflicting parent modifications; any conflict is explicitly recorded and
blocks activation. The v2 implementation and result artifact reflect this
component-level definition.

## Full materials benchmark Graph-RHI run

We ran the component-level Graph-RHI protocol on the complete
`materials_action_trajectories_v1` benchmark rather than the external fold
subset. The benchmark contains `20,987` action records and `45` grouped
trajectories. Trajectory-disjoint splitting produced `13,148` train, `2,648`
feedback, `2,622` acceptance, and `2,569` test actions. The fixed top-10%
action budget was used for every node.

| Harness | Top-10% risk | Hit rate | Mean utility | Acceptance | Active |
|---|---:|---:|---:|:---:|:---:|
| H0 fixed | 0.2996 | 0.7004 | 0.1387 | n/a | no |
| H1 evidence/source | 0.2957 | 0.7043 | 0.1471 | no | no |
| H2 OOD/stability | 0.2879 | 0.7121 | 0.1541 | yes | yes |
| H3 component merge | not activated | not activated | not activated | no | no |

The full machine-readable result is stored in
`runs/graph_rhi_full_v2/summary.json`. On the complete benchmark, H2 improves
over H0 by `0.0117` absolute risk and `0.0154` mean utility, while H1 is not
accepted. Since only one branch passes acceptance, the protocol correctly does
not activate a two-parent H3 merge.
