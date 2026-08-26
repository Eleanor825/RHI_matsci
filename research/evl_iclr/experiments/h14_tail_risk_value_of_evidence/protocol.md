# H14 Protocol: Tail-Risk Value of Evidence for Scientific Actions

## Status and contamination boundary

This protocol is locked before inspecting target distributions or running any
method on the external-confirmation datasets. H13 folds 0--4 are consumed and
may be used only for method development and diagnosis. They can never again be
described as untouched confirmation.

External confirmation uses the immutable snapshots in
`benchmarks/external_h14/manifest.json`:

- `matbench_dielectric` for pairwise preference;
- `matbench_jdft2d` for performance-plus-uniqueness discovery;
- `matbench_phonons` for extreme-property discovery.

The compressed source hashes in the manifest define the dataset versions.

## Falsified H13 assumptions

H13 ranks actions primarily by calibrated probability of positive gain. The
post-hoc fold-0 diagnostic shows that this is insufficient for asymmetric
scientific loss: the unique-discovery top 0.5% has 80% positive actions but
negative aggregate gain because a few false positives have large negative
value. H13 also acquires tools using a threshold on the base score, which
prevents a tool from rescuing candidates that the base model systematically
underranks.

## H14 hypothesis

A scientific harness should estimate the full asymmetric distribution of net
scientific gain and the value of acquiring reducible evidence. A
tail-risk-sensitive gain distribution and a learned counterfactual
value-of-evidence acquisition model will produce positive, risk-certified gain
on all three external tasks, while source-decomposed uncertainty will improve
selection over a matched no-source model.

## Strict net scientific gain

For state `s`, proposed action `a`, realized outcome `o`, and optional evidence
operation `t`, define

```text
G(a,o | s) = V(a,o) - c_exec(a) - h_fail(a,o) - rho_B(s)
```

where `V` is train-normalized continuous realized scientific utility, `c_exec`
is normalized execution cost, `h_fail` is additional irreversible failure harm,
and `rho_B` is the train-only opportunity cost induced by the fixed scientific
action budget `B=0.10`. Tool acquisition additionally pays `c_t`. A binary task
success label may determine failure harm or evaluation risk, but it does not
erase scientifically meaningful near-miss utility. No test outcome enters
normalization, reservation, model fitting, calibration, mutation, or threshold
selection.

The gain range is fixed before training to `[-1.40, 1.00]`, implied by the
normalization and cost contract. This bound is used for finite-sample gain
certificates and may not be changed after seeing external outcomes.

## Predictive distribution

H14 replaces the single Gaussian gain head with a hurdle distribution:

```text
q(G | x) = pi(x) q_plus(G | x, G>0) + (1-pi(x)) q_minus(G | x, G<=0).
```

Task-conditional cross-fitted models estimate:

- `pi = P(G>0 | x)`;
- positive-gain magnitude;
- negative-gain magnitude;
- calibrated component variances.

The harness reports numeric estimates only:

```text
mean_gain
p_positive_gain
expected_shortfall_beta
internal_variance
external_variance
tool_value_of_evidence[tool]
```

It does not output an execution route. The host agent policy consumes these
numbers under its cost and risk preferences.

## Internal/external source decomposition

For every action and evidence stage, predictions form a balanced matrix over
independently fitted model replicas `m` and independently generated evidence
views `e`. Holding evidence fixed identifies model-internal variance; averaging
over models and varying evidence identifies evidence-external variance through
the law of total variance:

```text
sigma_internal^2 = E_e Var_m[g(m,e)]
sigma_external^2 = Var_e E_m[g(m,e)].
```

External evidence views must differ through auditable source/tool operations,
not hand-edited confidence features. The matched no-source ablation receives
the same mean predictions, records, tool budget, and calibrator capacity but
sets both variance terms to zero.

## Tail-risk action score

For lower-tail level `beta=0.10`, define negative expected shortfall

```text
ES_beta(q) = E[-G | G <= Q_beta(G)].
```

The deployable scalar score used for ranking is

```text
W_lambda(q) = E_q[G] - lambda * ES_beta(q),
```

with `lambda` selected only from the predeclared grid
`{0.0, 0.25, 0.5, 1.0}` on feedback data. This score penalizes rare,
high-magnitude scientific failures that are invisible to `P(G>0)` alone.

## Counterfactual value of evidence

For tool `t`, let `D(q)=1[W_lambda(q)>0]` be the frozen execute/no-execute
decision induced by a predictive distribution. The realized counterfactual
evidence value on cross-fitted feedback trajectories is

```text
Delta_t = G * (D(q_t) - D(q_base)) - c_t.
```

A base-state acquisition model predicts `E[Delta_t | s]` without observing the
future tool result. Training targets are strictly out-of-fold: the same record
is never used to fit the tool posterior that defines its `Delta_t` target.
Acquisition is therefore based on predicted decision improvement, not on a
base-confidence threshold. The source-aware version may use reducible external
variance; internal variance is reported separately and is not assumed to be
reduced by an evidence tool.

## Harness evolution

Evolution occurs only on train and feedback data. A harness contract contains:

- visible signal families;
- evidence tools and their order;
- hurdle calibration family;
- `beta` and `lambda` tail-risk settings;
- value-of-evidence acquisition head;
- execution score and abstention contract.

Each generation receives action-level residual summaries and trajectory-level
net-gain summaries from feedback data, proposes bounded contract mutations, and
retains the best feedback candidate. Acceptance labels are never returned to
the mutator. After evolution stops, at most four frozen finalists are evaluated
once on the independent acceptance set.

## Selective deployment certificate

A task-conditional finalist and its numeric threshold are selected using only
feedback records. Acceptance is not searched: it evaluates that one frozen
policy once. The policy is deployable only if its one-sided exact upper
confidence bound on selective risk is at most `alpha=0.20`. Family-wise
`delta=0.05` is split across the three tasks. A policy selecting no acceptance
actions is not certified. If certification fails, that task uses the no-op
contract. The method, threshold, and tool contract are copied unchanged to
test.

Net gain is reported with all acquisition costs on every task-fold. It is not
used as a second per-fold deployment certificate: for low-coverage policies, a
distribution-free bounded-mean lower confidence bound has insufficient power
even when every selected action succeeds. Confirmatory gain evidence instead
comes from the predeclared cross-fold positive-gain counts and paired
randomization comparisons in the locked gates below. Selective risk remains
the finite-sample deployment guarantee.

## External benchmark construction

All transformations are fit separately within each outer fold.

### Pairwise preference

- Source: `matbench_dielectric`, target `n`.
- Proposed action: choose the candidate preferred by a train-only low-fidelity
  property model from a surrogate-matched pair.
- Hidden outcome: whether the proposed candidate has higher measured target.
- Utility: train-normalized absolute improvement over the alternative.

### Unique discovery

- Source: `matbench_jdft2d`, target `exfoliation_en`.
- Performance prefers lower exfoliation energy and is normalized using training
  records only.
- Uniqueness is distance from the training material library under a frozen
  structure/composition representation.
- Utility is the equal-weight performance/uniqueness score. Exact utility and
  advancement outcome remain hidden until evaluation.

### Extreme-property discovery

- Source: `matbench_phonons`, target `last phdos peak`.
- The action advances a candidate toward the high-tail extreme target.
- Utility is the train-normalized target percentile; the true target remains
  hidden until evaluation.

Outer folds are grouped by reduced chemical system so no chemical-system group
appears in both train and test. Five deterministic outer folds cover every
group exactly once. Within each outer-training partition, disjoint train,
feedback, and acceptance groups are assigned before fitting any model or
reference library.

## Baselines

Every baseline uses identical records, costs, action budget, outer folds, and
selective-risk evaluation:

- verbal confidence;
- evidence heuristic;
- ensemble lower confidence bound;
- static reliability calibrator;
- Gaussian continuous-gain head;
- hurdle gain without tail risk;
- tail risk without source decomposition;
- source decomposition without learned value of evidence;
- direct LLM judge from two GPT-family and two Claude-family models;
- hybrid LLM judge plus the same acceptance certificate.

## Locked confirmatory gates

The external experiment passes only if all corruption audits pass and:

- every active task-fold has test selective risk at most 0.20;
- each task has positive net gain and nonzero coverage in at least four of five
  outer folds;
- at least 12 of 15 task-folds have positive population net gain;
- mean paired population gain exceeds static reliability and direct LLM judge;
- the source-aware model has positive paired mean gain over the matched
  no-source ablation and a one-sided paired randomization `p < 0.05` across the
  15 task-fold cells;
- evolved contracts beat the frozen initial contract in paired mean gain while
  preserving both acceptance certificates;
- all thresholds and contracts transfer without external-test tuning.

Failure is retained. The gain bound, alpha, delta, coverage grid, tool costs,
task definitions, group folds, and statistical tests may not be changed after
external outcomes are inspected.

## Runtime requirement

External offline confirmation is necessary but not sufficient for the full
paper claim. A separate MatBot evaluation must log pre-action state, numeric
harness output, selected host action, tool provenance, costs, and delayed
outcome. No offline trajectory may be relabeled as a live MatBot trajectory.
