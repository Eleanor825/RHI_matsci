# Research Findings

## Research Question

Can a scientific-agent harness estimate the calibrated probability that a proposed action has positive net scientific gain, while separating uncertainty caused by the agent from uncertainty caused by scientific evidence and controlling the risk of accepted actions?

## Current Understanding

The publishable object should not be a routing policy. It should be a calibrated
numerical distribution over net scientific gain: expected gain, internal and
external variance contributions, positive-gain probability, and expected
shortfall. Internal and external uncertainty are not a flat feature list: they
are two operational sources of predictive variation obtained by resampling the
agent while holding evidence fixed, and resampling evidence/tool views while
averaging over the agent. Routing remains the responsibility of the host
MatBot harness.

## Validity Correction

- The earlier H5/H7/H8 five-seed results are controlled-replay evidence, not
  five distinct held-out-regime results. Matbench pairwise and unique discovery
  reused exactly the same test groups in every seed.
- The legacy budget reservation also depended on training success prevalence.
  In one split, a small prevalence change made the unique-task reservation jump
  from approximately 0.825 to zero.
- H12 repairs both issues with a stable potential-success slot value and five
  deterministic group-covering folds. Under this stronger protocol, fold 0 has
  zero certified coverage at the original acceptance allocation.
- H13 doubles only the independent acceptance allocation. Its development fold
  recovers 0.51% coverage, 3.33% test risk, and +0.000228 population net gain.
  Static reliability, ensemble LCB, verbal confidence, and evidence heuristic
  all remain at zero exact-risk-controlled coverage. This development result
  is excluded from confirmatory inference.
- Source-decomposed and no-source real-tool methods remain nearly identical;
  the development-fold gain difference is only +0.00000817. Internal/external
  decomposition is still a method hypothesis, not an established empirical
  contribution.

## Key Results

- H5.2 is the first preregistered positive controlled-replay result. On a
  leakage-audited sequential tool benchmark, task-stratified sparse continuous
  SD-Worth has nonzero exact-risk-controlled coverage, empirical risk at most
  0.20, and positive population net gain in all five legacy split seeds. Those
  seeds do not constitute five distinct regime tests.
- At `alpha=0.20` and normalized verification cost `0.002`, H5.2 averages
  1.43% execution coverage, 4.95% selective risk, 28.0% verification
  acquisition, and +0.001402 population net gain.
- Source decomposition beats the continuous no-source ablation in 3/5 seeds
  with mean paired gain +0.001353. The seed-bootstrap CI is positive, but the
  five-seed Wilcoxon test is not significant (`p=0.3125`), so this is a
  promising mechanism result rather than a final superiority claim.
- Continuous gain likelihood is essential: binary calibration can have low
  error rate while selecting negative mean gain because a few severe failures
  dominate many small successes.
- H8 supplies the first useful continuous-loss certificate. With the same
  normalized shortfall loss, scores, splits, and coverage grid, the Hoeffding
  certificate gives nonzero coverage, positive selected gain, and positive
  population net gain in 4/5 seeds. The original empirical-Bernstein bound
  remains at zero coverage in 5/5 seeds because its finite-sample penalty is
  too loose at the current acceptance-set size.
- Hoeffding-certified recursive harness evolution is also positive with
  nonzero coverage in 4/5 seeds. In the remaining seed it correctly retains
  the zero-action incumbent rather than paying verification cost for an
  uncertified contract.
- Continuous shortfall and binary failure risk are different claims. H8 keeps
  held-out mean normalized shortfall below 0.20 in all five seeds, while the
  binary failure rate can exceed 0.20 because many failures have small
  negative gain. The paper must report both rather than call either one simply
  “risk.”
- H7 is the first positive controlled recursive harness-evolution result. Under a
  family-wise exact-risk correction over five harness contracts and eight
  coverage thresholds, acceptance-set evolution produces nonzero coverage,
  risk at most 0.20, and positive population net gain in all five seeds.
- H7 selects four different final contracts across five legacy splits, confirming
  that evolution occurs at the harness-contract level rather than once per
  action. Mean net gain is 0.003043 and median net gain is 0.000538.
- The fixed source-decomposed harness is also positive in all five seeds and
  has a slightly higher median; H7's higher mean comes from accepting a strong
  sparse-verification contract in seed 13. This supports adaptive contract
  selection but is not yet a universal superiority result.
- H6.1 provides the first real GPT-5.6 model-replica evidence on the v2 tool
  contract. In a corrected 54-cell, worthiness-balanced diagnostic, raw direct
  judging is poor (Brier 0.3681, ECE 0.4919, accuracy 0.50), while internal and
  external gain variances correlate with absolute gain error at 0.8667 and
  0.6766 respectively.
- This supports using LLM replicas as uncertainty axes inside a calibrated
  gain model, not using raw LLM confidence as the final action-worthiness
  score. The result is exploratory because only six actions were deliberately
  label-balanced.

- H1.1 makes action worthiness distinct from success: on seed 7, 22.7% of test labels differ after adding the train-only action-budget reservation value. However, current pooled heads still have zero finite-sample risk-controlled coverage through alpha 0.3.
- H1.2 confirms on four untouched seeds that task-conditioned static heads improve calibration but not strict execution: mean Brier falls to 0.0934 and AURC to 0.8268, while risk-controlled coverage remains zero for every method through alpha 0.3.
- The balanced H2 software pilot cleanly detects aligned evidence interventions versus permuted controls, but the resulting variances do not predict held-out gain error. This separates a correct variance-decomposition implementation from a valid scientific uncertainty signal.
- A complete 270-cell closed-model/tool pilot shows that train-library tool views improve calibration but not ranking. Raw GPT probabilities are badly task-incomparable: Matbench probabilities dominate while unique/extreme probabilities collapse near zero, motivating mandatory task-conditional calibration.
- Nonlinear tree ensembles do not establish non-zero strict coverage, ruling out linear model capacity as the main explanation for H1 failure.
- Candidate-specific simulation helps ranking modestly, but the LLM ensemble dilutes the simulator signal; the scientific surrogate should be the primary evidence model and LLMs should contribute one internal/model-uncertainty axis.
- The only H3 non-zero result was a discrete-quantile artifact. A zero-gain outside option is required in the reservation definition; after correction, all current static methods return zero guaranteed coverage.
- The sanitized benchmark removes legitimate task-defining database/simulation outputs. Future experiments must model them as costly sequential tool calls rather than hide them completely or leak them into the base prompt.
- The current historical benchmark is internally consistent and leakage-audited, but its labels/utilities remain benchmark-derived proxies.
- Static calibration/fusion improves a cached GPT-5.5 direct-judge score on a 500-record subset, but this is not a full discovery-utility or RHI result.
- The current five-seed recursive self-evolution mechanism fails to improve the aggregate primary score over H0; this rules out using the present RHI mutation scheme as the main positive claim.
- Prompt-conditioned trajectories are deterministic construction traces, not live MatBot/tool trajectories.

## Patterns and Insights

- Strong pilot gains can persist after permuting weak-model internal signals, indicating that local benchmark features rather than model uncertainty caused the gain.
- A feature being listed in a harness is not evidence it was measured; missing signals were previously represented by constants.
- Source decomposition is not sufficient by itself: evidence views must be generated by real retrieval/tool variation. Hand-modifying `evidence_support` or injecting generic conflict produces identifiable but scientifically meaningless variance.
- Scientific action utility, failure harm, and action cost must be separated before calibration; otherwise the target is difficult to interpret across tasks.
- A publishable model should treat source variances as components of a heteroscedastic gain distribution, not concatenate them as arbitrary classifier features.
- Sparse evidence acquisition is part of the method, not an engineering
  afterthought: full verification can improve ranking while destroying net
  gain once tool cost is charged.
- Task-specific acquisition thresholds improve cost efficiency, while pooled
  acceptance certification retains statistical power. Task-wise Bonferroni
  certificates were too conservative at the current sample sizes.
- Exact risk control is currently power-limited in the narrow useful region:
  H12 fold 0 has 0/19 errors at top 1% but UCB 0.2344. Doubling acceptance in
  H13 lowers the accepted-prefix UCB to 0.1831 without relaxing alpha or the
  coverage-grid correction.
- More acceptance data does not repair broad ranking: in H12 the 5% prefix has
  29.67% empirical risk. Scientific ranking quality and statistical power are
  separate bottlenecks.
- H13 confirms that powered acceptance can recover a narrow safe region under
  true held-out regimes: three of four untouched folds have positive net gain,
  all active-fold risks remain below 0.20, and the mean gain exceeds all four
  static baselines under the same accounting.
- H13 also falsifies the broad current claim. All selected actions are
  Matbench pairwise, and the source-decomposed calibrator improves mean
  population gain over no-source by only 0.000007 with one win in four folds.
  The next method must improve task-specific scientific evidence and make
  source variation identifiable; additional acceptance tuning is not enough.

## Lessons and Constraints

- Never describe the current data as full internal/external uncertainty.
- Never use test regimes for utility normalization, calibration, harness selection, or risk-threshold selection.
- The next positive result must beat static reliability and direct LLM judge under identical action budgets and held-out regimes.
- H5.2 and H8 pass their controlled-benchmark criteria but do not satisfy the
  full project goal because the source-ablation advantage is not statistically
  decisive, one H8 seed has zero certified shortfall coverage, and trajectories
  are not live MatBot.
- Real MatBot/tool trajectories remain a hard requirement, not an optional extension.

## Open Questions

- Which train-only normalization preserves meaningful cross-task net gain?
- Which task-specific evidence operation creates positive oracle headroom for
  unique and extreme discovery without leaking benchmark outcomes?
- Which controlled but scientifically valid interventions separately vary
  model-internal and evidence-external uncertainty enough to identify their
  marginal decision value?
- Does the H5.2 advantage persist across open models, closed GPT/Claude judges,
  and live MatBot trajectories?
- Does task-conditional calibration turn the strong H6.1 variance/error signal
  into improved risk-controlled utility on a naturally sampled complete API
  matrix?
- Can internal/external variance components be estimated reliably with affordable model/tool replicas?
- Which real MatBot actions have auditable delayed outcomes?

## Optimization Trajectory

H1 was refuted because gain positivity collapsed to success. H1.1 introduced a
budget-conditioned reservation but later auditing found a prevalence
discontinuity. H5/H7/H8 established controlled-replay feasibility, H10 showed
that real scientific tools transfer only weakly, and H12 invalidated the old
repeated-regime evaluation. H13 transferred a narrow safe positive-gain region
to three of four untouched folds and beat static baselines, but failed its
locked multi-task and source-ablation gates. The next phase requires new
external confirmation, task-specific evidence, and identifiable source
variation rather than further tuning on the consumed H13 folds.

H14 fold-0 development identifies where source decomposition is useful. When
all sources are fused before execution, a dominant high-fidelity source makes
source-aware and no-source policies nearly identical. When evidence-external
variance is instead used before acquisition to predict counterfactual tool
value, the source-aware policy becomes independently deployable on extreme
discovery while the matched no-source policy does not. This supports the
specific claim that external uncertainty is an acquisition signal, not a
generic confidence penalty. The result is development-only and currently
holds for one of three tasks. Pairwise transfer instability and unique-task
no-op show that feedback-learned native-margin calibration remains the main
bottleneck; the next estimator must transform source outputs into train-only
scientific utility units before learning residual calibration.

## H17 confirmatory finding: tail probability is more stable than mean VoE

The preregistered mean-VoE primary did not survive chemical-system shifts. Its
2/4 certification rate and lower mean deployed gain than the matched no-source
ablation falsify the strong expected-value claim. In contrast, the auxiliary
probability that evidence has positive value certified 3/4 folds and improved
mean deployed gain by 35.5% over its matched ablation. This suggests that, under
sparse heavy-tailed scientific gains, acquisition should be modeled as a
hurdle/tail event rather than by a noisy conditional mean alone. This hypothesis
requires a new locked external test and cannot be declared confirmed from H17.

## H16/H17 benchmark finding: uncertainty needs identifiable evidence states

A benchmark containing only uniformly ambiguous nearest-neighbor pairs can make
costed acquisition impossible even when the post-acquisition tool is excellent.
A scientifically meaningful routing benchmark must contain distinguishable
states in which internal confidence, cross-source disagreement, and OOD evidence
have different consequences. H17 creates those states using train-only models
without conditioning pair selection on hidden outcomes.

## Runtime evidence status

The MatXBot logging and import path is implemented with a hard separation
between decision-time events and observed outcomes. Three genuine OpenClaw
turns have been captured from the real high-energy-density-battery workspace:
one GPT-5.6 Sol and two independent GPT-5.6 Luna runs, totaling 59 tool calls.
The Luna replicas proposed substantially different experiments (token Jaccard
`0.1791`) while self-reporting nearly identical internal uncertainty (`0.43`
and `0.42`). This establishes that self-reported uncertainty can miss action
instability, but the live turns still lack post-experiment outcomes.

Separately, 12 real historical LFP A/B experiments now form complete offline
replay trajectories with measured terminal DCR/cycle outcomes. They validate
the post-outcome join and utility path. Because the source was inspected before
construction and contains only 12 records, it is development evidence rather
than a held-out paper result.

## Contract finding: verification removes unstable base-action risk

The main H17 failure came from executing some actions without acquiring the
high-fidelity observation. A verify-before-execute contract certified all five
consumed development folds and increased stable net gain, while static external
variance activated only once. The result supports evolving the action contract,
not merely its prompt or threshold. Because the same folds informed the change,
this is a development finding and requires a new external dataset.

## External confirmation: learned VoE beats static uncertainty, not its ablation

Across four untouched dielectric folds, verify-before-execute learned VoE
certified every fold and produced positive gain, while static external variance
never activated. However, source-aware positive-VoE did not improve aggregate
gain over the matched no-source learner. The defensible current claim is that
cost-aware counterfactual VoE and contract-level verification matter; source
decomposition provides diagnostic structure but does not yet earn an average
performance claim.

## LLM-judge finding: success confidence is not evidence value

On the same H18 actions, GPT-5.6 judges were well calibrated for ordinary action
success yet poorly aligned with the rare event that high-fidelity evidence has
positive counterfactual net value. Their top-budget acquisitions lost net gain
after tool cost. The learned VoE score had worse action-label calibration—as
expected because it estimates a different quantity—but much higher positive-VoE
ranking quality and positive net gain. This distinction is a candidate central
paper result, pending a larger risk-certified model comparison.
# Current falsifiable findings (2026-08-22)

1. **The robust contribution is value-of-evidence under selective risk, not raw source disagreement.** H17/H18 show positive cost-aware VoE gains, while H19 falsifies a broad claim that source-aware features reliably beat a matched no-source estimator on every new property task.
2. **Confidence of success is not equivalent to evidence worth.** On the locked H18 risk-500 subset, our VoE policy certified and deployed at `+0.000563`; GPT-5.6 Luna and Sol direct judges did not obtain enough certified execution coverage and therefore deployed at zero gain.
3. **Internal/external variance requires an interaction term.** The previous conditional-variance definition mixed replica-by-source interaction into the internal component. The implementation now uses an exact balanced ANOVA decomposition into internal, external, and interaction variance.
4. **Budget rank routing is not a universal fix.** It stabilizes coverage under score shifts but reduced H17 certification from 5/5 to 4/5 and did not beat no-source.
5. **Per-tool numeric VoE is safer than a learned route label, but cheap tools are not yet useful.** Familywise contract evolution safely falls back to high-fidelity. H17 and all five frozen H20 folds selected only high-fidelity; H20 source-aware gain was `+0.001390` versus no-source `+0.001410`, while static reliability certified 0/5. A paper-level multi-tool claim still requires cheaper certified tools to improve net gain.
6. **Model self-report does not measure action instability.** Across two independent Luna turns, reported internal uncertainty had range `0.01`, but action token Jaccard was `0.1791` and the pre-outcome net-gain proxy had range `0.2698`. This motivates distributional internal uncertainty over repeated scientific actions rather than a verbal scalar.
7. **Real MatBot evidence now exists on both sides of the outcome boundary, but not in one confirmatory dataset.** Three live runtime trajectories preserve real tool use but lack outcomes; 12 historical LFP replay trajectories have real outcomes but are small and development-only. A held-out runtime-plus-outcome campaign remains necessary.
8. **Scalar sign entropy is not sufficient distributional uncertainty.** The D3 boundary term slightly improved mean Brier/ECE on consumed H17 folds but worsened AURC and log loss on all five, and reduced fold-0 VoE gain. The method must model actual action choices/proposals, not only repeated scores for a fixed action.
9. **A real action-distribution replay is now available.** The 12 measured LFP experiments yield 19 within-protocol A/B decisions with outcome-blind orientation and complete hidden-outcome trajectories. This enables repeated-model choice entropy to be compared directly with self-reported uncertainty and realized scientific utility.

## 2026-08-22: Action instability is not itself calibrated uncertainty

On 19 real historical LFP pairwise actions, three GPT-5.6 variants with three
replicas produced nonzero choice entropy on 42.1% of records, confirming that
scientific action selection is unstable. However, action entropy had only
0.0117 correlation with direct-probability error. A frozen leave-one-batch-out
entropy-shrinkage model worsened Brier from 0.19017 to 0.27136 and AURC from
0.30734 to 0.57091. Self-reported uncertainty failed similarly. Therefore the
paper cannot claim that disagreement is automatically uncertainty. The viable
hypothesis is narrower and testable: crossed action distributions expose
internal, external, and interaction variance components that may improve a
separately calibrated net-gain distribution.

## 2026-08-23: Source decomposition helps conditionally, not universally

H23 provides the first untouched extreme-property task on which the frozen
source-aware primary passes every predeclared gate. Across four confirmatory
chemical-system folds, it certifies in three, executes 396 actions with zero
errors, and improves pooled population gain from `+0.000580` without source
features to `+0.000692` with them. Fixed-sequence evolution nevertheless keeps
source features in only one deployed fold. Together with the H18--H20 negative
transfers, the defensible claim is that recursive acceptance can retain source
decomposition where it improves certified net gain and prune it elsewhere;
source disagreement is not a universally beneficial score.

## 2026-08-23: Same-evidence LLM advantage is positive but underpowered at 485

On the frozen balanced subset, portfolio KG gains `32.2719` versus `28.0575`
for the best-gain GPT-5.6 ensemble and `26.7116` for the only risk-certified
ensemble. The numerical harness has one error in 49 executions and passes the
`alpha=0.10` certificate. However, paired resampling does not yet establish a
5% significant gain advantage. The paper must report this as an underpowered
positive effect until the predeclared 2,538-record extension finishes.

## 2026-08-23: Full same-evidence comparison establishes LLM superiority gap

The predeclared 2,538-record extension reverses the subset's power limitation.
Frozen portfolio KG gains `195.7886`, while the strongest frozen GPT-5.6
screening ensemble gains `153.4501` and does not satisfy the 10% CP risk
certificate. The paired gain difference is `+42.3385`, with benchmark-
stratified bootstrap interval `[22.1953, 62.5509]` and one-sided sign-flip
`p=0.00002`. The numerical harness also lowers empirical selective risk from
`9.84%` to `0.79%`. This supports the central claim that calibrated numerical
action valuation and portfolio evidence value outperform direct LLM judging
under the same evidence and action budget.

## 2026-08-23: Portfolio value beats indiscriminate evidence, not decisively no evidence

On the full 2,538-record acceptance set, portfolio KG improves strict gain by
`+27.3906` over fixed full-evidence acquisition; the paired bootstrap interval
is `[10.6264, 43.9305]` and one-sided sign-flip `p=0.00065`. Against the strong
base-only factorized LCB, the gain is `+6.9968`, but the interval
`[-10.2965, 24.3929]` includes zero and `p=0.217`. This falsifies a broad claim
that evidence acquisition is always necessary. The supported claim is that
learned portfolio value prevents the loss caused by indiscriminate evidence
while preserving a positive, currently non-significant advantage over using no
additional evidence.

## 2026-08-23: Risk-certified LLM judging remains substantially lower value

The full verification-stage GPT-5.6 ensemble is safer than the screening
ensemble—10/254 errors, empirical risk `3.94%`, and CP UCB `6.59%`—but its
evidence cost reduces strict trajectory gain to `141.0945`. EVL remains both
safer and more valuable: gain `195.7886`, 2/254 errors, and paired advantage
`+54.6940` with bootstrap interval `[31.0198, 78.5772]` and one-sided
`p=0.00001`. Direct LLM judging therefore does not close the gap merely by
receiving more evidence.
