# SD-Worth Method Specification

## 1. Scientific Decision Object

At runtime, a host scientific agent proposes an action `a` in state `s`.
SD-Worth does not choose a route and does not replace the host policy. It
returns a numerical gain-distribution estimate:

```text
mu_G(s,a)                 calibrated expected net scientific gain
sigma_internal^2(s,a)    calibrated model/agent variance contribution
sigma_external^2(s,a)    calibrated evidence/tool variance contribution
q(s,a)                    P(G(s,a; B) > 0 | pre-action information)
ES_shortfall(s,a)         E[max(-G(s,a; B), 0) | pre-action information]
```

`q` remains the scalar action-worthiness score used for selective-risk
calibration. Expected gain and expected shortfall prevent the host from
treating equal positive-gain probabilities as equal when their scientific
upside or downside magnitudes differ. A downstream MatBot budget policy can
consume these numbers, but route names such as `execute`, `simulate`, or
`ask_expert` are deliberately outside the estimator contract.

## 2. Strict Net Scientific Gain

Utility normalization is fit on the training split only and separately within each scientific task. The outcome-conditioned base gain is:

```text
base_gain = success * normalized_scientific_utility
            - lambda_cost * normalized_action_cost
            - lambda_failure * (1-success) * irreversible_failure_harm
```

The legacy controlled experiments used the train-only quantile of
outcome-conditioned base gain. H12 replaces it because this quantity jumps
discontinuously when the training success prevalence crosses `B`. For action
budget fraction `B`, define the potential successful slot value:

```text
slot_value = normalized_scientific_utility
             - lambda_cost * normalized_action_cost

reservation_task(B) = max(0, quantile_(1-B)(slot_value on task training data))
```

The supervised target is:

```text
G(s,a;B) = base_gain - reservation_task(B)
worthiness = 1[G(s,a;B) > 0]
```

This target differs from success: a successful but low-value action can be
unworthy because it consumes a slot that could be spent on a better candidate.
The reservation is fit from training outcomes only, but is not multiplied by
the binary success label and therefore remains stable when success prevalence
changes slightly across source partitions.

## 3. Operational Source Decomposition

For every audited action, collect a balanced matrix of gain predictions `g[m,e]`:

- `m` indexes agent/model replicas while evidence is held fixed;
- `e` indexes independently generated evidence/tool views while the agent replica is held fixed.

The matrix must contain every agent-by-evidence cell. Missing cells are not imputed.

```text
mu          = mean_(m,e) g[m,e]
U_internal  = mean_e Var_m(g[m,e])
U_external  = Var_e mean_m(g[m,e])
U_total     = U_internal + U_external
```

For open models, the agent axis can include sampling replicas plus answer-logit difference, token-level predictive entropy, semantic sample disagreement, and hidden-state confidence probes. For closed models, the black-box substitute is repeated/model-family sampling with identical prompts, supported by self-reported assessment confidence only as an auxiliary signal. Verbal confidence alone is not treated as intrinsic uncertainty.

The evidence axis must be produced by auditable retrieval, simulation, property-prediction, database, or expert tools. Manually changing an `evidence_support` feature is a software intervention, not valid scientific evidence.

## 4. Source-Decomposed Gain Distribution

SD-Worth models budget-conditioned gain as a heteroscedastic distribution:

```text
G | s,a ~ Normal(
    a_task * mu + b_task,
    sigma_task^2
    + lambda_internal_task * U_internal
    + lambda_external_task * U_external
)
```

with `a_task > 0` and non-negative source weights. Parameters are fit by
regularized Gaussian negative log likelihood on continuous budget-conditioned
gain, rather than binary cross entropy alone. The calibrated numerical output
contains:

```text
mu_G              = a_task * mu + b_task
sigma_internal^2  = lambda_internal_task * U_internal
sigma_external^2  = lambda_external_task * U_external
sigma_total^2     = sigma_task^2 + sigma_internal^2 + sigma_external^2
q                 = Phi(mu_G / sigma_total)
ES_shortfall      = sigma_total * phi(mu_G / sigma_total)
                    - mu_G * Phi(-mu_G / sigma_total)
```

Task-conditional parameters are fit on an independent feedback/calibration split. This is necessary because raw model probabilities and gain scales differ substantially across scientific tasks. Once calibrated, every task returns the same semantic quantity: probability of positive net gain.

## 5. Sparse Evidence Acquisition

The host policy does not call an expensive verification tool for every action.
On the feedback split, it learns one base-score acquisition threshold per
scientific task under a declared tool cost. At runtime:

```text
base score below acquisition threshold -> no verification call
base score above acquisition threshold -> acquire independent tool views
                                      -> compute gain-distribution numbers
```

This preserves a numerical uncertainty-harness interface while allowing the
host to allocate evidence cost differently across tasks. The acquisition
policy is frozen before acceptance and test evaluation.

## 6. Selective Risk Control

On an independent acceptance split, predeclare candidate coverage fractions. For each score-ranked prefix, compute a one-sided exact binomial upper confidence bound on:

```text
selective_risk = P(G <= 0 | action accepted)
```

Apply Bonferroni correction across the finite coverage grid and select the largest coverage whose upper bound is at most `alpha`. The guarantee assumes exchangeability between acceptance and deployment actions. Held-out scientific regimes are reported as empirical transfer tests and are not mislabeled as distribution-free guarantees.

Because binary error rate does not control the severity of negative actions,
SD-Worth also reports a bounded utility-shortfall loss:

```text
L_shortfall = min(1, max(0, -G / train_only_scale_task))
```

Two distribution-free upper bounds are evaluated over the same finite coverage
grid. The original empirical-Bernstein bound remains valid but has zero useful
coverage at the current acceptance-set size. H8 therefore preregisters a
Hoeffding bound for the unchanged bounded shortfall loss:

```text
E[L_shortfall] <= mean(L_shortfall)
                  + sqrt(log(1 / delta') / (2 n))
```

where `delta'` is Bonferroni-corrected across all candidate coverages and,
during harness evolution, across all candidate contracts. The shortfall
certificate controls expected severity of negative gain; it does not imply a
bound on the binary probability `P(G <= 0)`. Binary selective risk and
continuous shortfall are therefore reported in separate columns.

## 7. Harness Self-Improvement

Recursive harness improvement operates at the contract level, not once per action. A candidate harness can change:

- which internal replicas are requested;
- which external tools or evidence views are acquired;
- the source-decomposed calibration parameters;
- feature extraction and cost budgets;
- prompt contracts used to obtain numerical gain predictions.

The agent's scientific action policy is not rewritten by SD-Worth. Harness
evolution operates over a sequence of complete contracts, not once per action.
The current H7 candidate sequence adds, replaces, or removes:

1. sparse verification;
2. direct tool-probability judging;
3. task-specific acquisition thresholds;
4. continuous gain calibration;
5. source-decomposed variance calibration.

For `H` candidate harnesses and `K` candidate coverages, risk confidence is
corrected over all `H x K` choices. A mutation is accepted only when it has
nonzero certified coverage and higher acceptance-set population net gain than
the incumbent. The final contract and threshold are frozen before test.

The evolution procedure is run separately for the binary-failure certificate
and the continuous-shortfall certificate. Because the confidence statement is
simultaneous over the finite candidate set, choosing the highest-gain certified
contract on the acceptance split does not invalidate the corresponding risk
bound under the stated exchangeability assumption.

On the controlled v2 benchmark, H7 selects four different final contracts
across five split seeds and obtains positive risk-controlled population net
gain in all five. This replaces the repository's earlier negative RHI result,
but only for controlled replay; live MatBot self-improvement remains untested.

## 8. Current Controlled-Benchmark Evidence

On the leakage-audited sequential benchmark v2, H5.2 uses task-specific gain
heads, complete 3 x 3 model/tool replica matrices, continuous source
calibration, task-stratified sparse acquisition, and a pooled exact acceptance
certificate. Across five held-out-regime seeds at `alpha=0.20` and normalized
verification cost `0.002`:

- nonzero exact-risk-controlled coverage: 5/5 seeds;
- empirical test risk at most 0.20: 5/5 seeds;
- positive population net gain: 5/5 seeds;
- source-decomposed method beats the no-source continuous ablation: 3/5 seeds;
- mean paired population-gain difference: +0.001353;
- source weights are nonzero on at least two tasks in every seed.

The five-seed Wilcoxon comparison is not significant (`p=0.3125`), and the
benchmark is semi-synthetic controlled replay rather than live MatBot. These
limitations prohibit a final ICLR-level claim at this stage.

H7 recursive acceptance improves mean population net gain to `0.003043` with
mean selective risk `0.0304`; every seed has nonzero coverage and positive net
gain. Its fixed source-decomposed candidate remains more robust in median
performance, so H7 supports adaptive contract selection rather than uniform
dominance over every fixed harness.

H8 replaces only the concentration bound, leaving the gain target, bounded
shortfall loss, model, data splits, evidence-acquisition policy, and coverage
grid unchanged. At `alpha=0.20`, the fixed source-decomposed harness has
nonzero coverage, positive selected gain, and positive population net gain in
4/5 seeds. The Hoeffding-certified recursive harness also has nonzero coverage
and positive population net gain in 4/5 seeds; held-out mean shortfall is below
0.20 in every seed. The empirical-Bernstein certificate remains at zero
coverage in all five seeds. Thus H8 establishes a useful continuous-loss
certificate at the current sample size, while one split seed still falls back
to the zero-action incumbent.

## 9. Required Final Evidence

A paper-level positive claim requires:

1. non-zero risk-controlled coverage and positive net gain;
2. improvement over static reliability and direct LLM judges;
3. source-specific intervention tests and shuffled-axis controls;
4. real retrieval/simulation/tool views and live MatBot-compatible trajectories;
5. open and closed model families;
6. three scientific tasks and held-out scientific regimes;
7. ablations for internal variance, external variance, task calibration, reservation value, and risk control.
