# Research Log

| # | Date | Phase | Entry |
|---:|---|---|---|
| 1 | 2026-08-21 | bootstrap | Initialized autonomous research state for the seven-part ICLR objective. No completion claim is allowed until every requirement has direct evidence. |
| 2 | 2026-08-21 | audit | Existing repository evidence: 20,987 action records across three tasks and 45 regimes; 900 deterministic prompt-conditioned trajectories; zero verified live MatBot/tool trajectories. |
| 3 | 2026-08-21 | audit | Current full uncertainty signal contract fails: only verbal confidence and proxy perturbation stability have full internal coverage; tool agreement and answer-logit fields are absent. |
| 4 | 2026-08-21 | audit | Existing five-seed self-evolution ablation does not improve the aggregate primary score over H0. Current RHI is therefore a negative result, not evidence of self-improvement. |
| 5 | 2026-08-21 | direction | Locked working method direction SD-Worth: predict positive net scientific gain, decompose predictive variation by fixed-evidence agent replicas versus fixed-agent evidence replicas, calibrate worthiness, and select actions with a finite-sample selective-risk upper bound. |
| 6 | 2026-08-21 | inner-loop | H1 seed-7 sanity refuted: gain positivity matched success on all 20,987-derived split records, so binary worthiness was identical to static reliability. Exact all-prefix risk control at alpha 0.1 yielded zero coverage. |
| 7 | 2026-08-21 | direction | Locked H1.1 budget-conditioned opportunity-cost target and finite predeclared coverage-grid risk control. |
| 8 | 2026-08-21 | inner-loop | H1.1 seed-7 development run broke the success-label equivalence (22.7% test mismatch), but all methods retained zero coverage through alpha 0.3. Top-1% acceptance empirical risk was 61.9% for static reliability, 52.4% for binary worthiness, and 47.6% for the gain model. |
| 9 | 2026-08-21 | audit | Seed-7 visible-feature audit found strong task heterogeneity: the best univariate top-1% risk was 25% for Matbench pairwise, 33% for extreme properties, and about 73% for unique-material discovery. |
| 10 | 2026-08-21 | direction | Locked H1.2 task-conditioned heads. Seed 7 is development-only; seeds 1, 13, 21, and 42 are reserved for confirmation. |
| 11 | 2026-08-21 | confirmatory | H1.2 was refuted on untouched seeds 1/13/21/42. All five static-feature methods selected zero actions for alpha <= 0.3. Task-conditioned binary heads improved mean Brier from 0.0951 to 0.0934 and AURC from 0.8357 to 0.8268, but did not establish useful risk-controlled execution. |
| 12 | 2026-08-21 | decision | Static feature tuning stops here. The next model must use measured agent/evidence replicates and operational source decomposition; otherwise the strict selective-risk objective is not identifiable from current inputs. |
| 13 | 2026-08-21 | h2-pilot | Balanced 3x3 replica pilot completed on 3,113 held-out actions. Observed external variance exceeded independently permuted-view variance in 97.38% of controls, validating the decomposition implementation. |
| 14 | 2026-08-21 | h2-pilot | The controlled evidence proxies are not scientifically predictive: external-variance/error correlation was 0.027 and internal-variance/error correlation was -0.137; both dropout and conflict interventions increased predicted gain. These proxies are rejected as final uncertainty signals. |
| 15 | 2026-08-21 | h2-real | Completed a real 3-model x 3-tool-view matrix: 30 held-out actions and 270 cached cells from GPT-5.6 sol/luna/terra with base evidence, train-only nearest analogs, and train-only task statistics. No cells were imputed. |
| 16 | 2026-08-21 | h2-real | Tool-view averaging improved Brier 0.2033 to 0.1654 and ECE 0.2261 to 0.1822, but AURC slightly worsened 0.8643 to 0.8654. The improvement is primarily prior shrinkage, not better ranking. |
| 17 | 2026-08-21 | capacity-audit | ExtraTrees and histogram boosting also produced zero exact risk-controlled coverage through alpha 0.3 across five seeds. Nonlinearity improves AURC modestly but does not rescue the static feature contract. |
| 18 | 2026-08-21 | method | Implemented the constrained heteroscedastic source calibrator: task-conditional positive-gain probability with separate non-negative internal and external variance weights. |
| 19 | 2026-08-21 | trajectories | Exported 30 complete real-API/offline-tool replay trajectories. The schema separates visible state, tool calls, model readouts, scalar harness output, and hidden evaluation outcomes; it explicitly records online=false and matbot_runtime=false. |
| 20 | 2026-08-21 | h2-tool | Candidate-specific simulator evidence improved the 30-action GPT ensemble AURC from 0.8643 base / 0.8654 task-prior to 0.8541 and Brier to 0.1599, but top-budget gain remained negative. The simulator alone ranked better (AURC 0.7918), so LLMs should be replicas rather than the primary judge. |
| 21 | 2026-08-21 | h3-audit | Utility-shortfall H3 v1 produced a one-seed positive artifact because a discrete reservation quantile equaled the failed-action mass point. Adding the zero-gain abstention outside option removed the artifact; H3.1 returned zero coverage on all five seeds. |
| 22 | 2026-08-21 | benchmark-audit | Raw benchmark inspection found that the historical converter removes task-defining pre-execution tool outputs: Matbench property evidence for unique-material screening and generated-property evaluations for RL-CC. Current sanitized states are therefore under-informative for strict worthiness. |

Protocol locking uses timestamped protocol files plus SHA-256 digests because repository commits are not authorized in this session.
# 2026-08-21 — H4 sequential benchmark seed-7 sanity

- Locked and implemented a leakage-audited semi-synthetic sequential tool
  benchmark for all three material-discovery tasks.
- Verification evidence restored nonzero exact risk-controlled coverage and
  materially improved calibration, but H4 failed trajectory net gain because
  the shared gain head under-ranked utility magnitude and verification was
  charged to every action.
- Locked H4.1 before execution: task-specific gain heads plus feedback-selected
  sparse verification acquisition, with a predeclared cost-sensitivity curve.

# 2026-08-21 — H5.2 five-seed confirmatory result

- Replaced binary-only source calibration with a continuous Gaussian gain
  likelihood and added exact binary plus bounded-shortfall certificates.
- Added task-stratified sparse verification acquisition and retained a pooled
  acceptance certificate after task-wise Bonferroni thresholds had zero power.
- H5.2 passed all preregistered primary criteria: 5/5 nonzero coverage, 5/5
  risk <= 0.20, 5/5 positive population net gain, and 3/5 wins over the
  no-source continuous ablation.
- Mean paired net-gain improvement is positive (+0.001353), but the five-seed
  Wilcoxon result is not significant (p=0.3125). Bounded-shortfall coverage is
  zero and remains a negative result.

# 2026-08-21 — Sequential benchmark v2 trajectory release

- Exported 900 leakage-audited trajectories: 300 each for pairwise materials
  preference, unique-material discovery, and extreme-property discovery.
- Every trajectory contains base, screening, and verification observations,
  explicit incremental/cumulative tool costs, and provenance.
- Hidden labels, utilities, gain targets, and latent task margins are stored in
  a separate evaluation-only object. The manifest explicitly marks the data as
  semi-synthetic controlled replay and not live MatBot runtime.

# 2026-08-21 — H7 risk-controlled recursive harness evolution

- Added five harness contracts to a recursive mutation sequence and corrected
  risk confidence family-wise across all candidate-contract and coverage
  choices.
- A mutation is accepted only when it has certified nonzero coverage and
  improves acceptance-set population net gain over the incumbent.
- H7 passes all preregistered criteria across five seeds: nonzero coverage,
  empirical risk <= 0.20, and positive population net gain in every seed.
- Final contracts differ across regimes, providing controlled-benchmark
  evidence for harness-level self-evolution. The result is not yet validated
  on live MatBot or multiple model families.

# 2026-08-21 — H6.1 balanced-diagnostic implementation audit

- The first exploratory balanced API run sampled by raw success label, not by
  budget-conditioned action worthiness. The observed 4:2 class count violated
  the locked one-positive/one-negative-per-task contract.
- Marked the run invalid for discrimination analysis, locked a correction, and
  required exact 3:3 worthiness balance before rerunning.

# 2026-08-21 — H6.1 corrected real-model diagnostic

- Completed a 54-cell matrix using GPT-5.6 Sol/Luna/Terra and three independent
  verification observations for six worthiness-balanced held-out actions.
- Raw direct-judge probabilities were poorly calibrated and achieved chance
  threshold accuracy.
- Internal and external replica variance correlated strongly with absolute
  gain error (0.8667 and 0.6766), supporting source decomposition as a
  calibration input rather than raw confidence as a final score.
- Kept the result exploratory because label-balanced sampling uses evaluation
  labels and the natural 30-action matrix remains incomplete.

# 2026-08-21 — H8 Hoeffding-certified utility shortfall

- Kept the H5.2/H7 gain target, train-only shortfall scale, score models,
  acquisition policies, split seeds, and eight-point coverage grid unchanged.
- Replaced only the loose empirical-Bernstein shortfall upper bound with a
  preregistered Hoeffding upper bound, Bonferroni-corrected across coverage
  choices and across harness contracts during evolution.
- The fixed source-decomposed harness achieves nonzero coverage, positive
  selected gain, and positive population net gain in 4/5 seeds. Held-out mean
  normalized shortfall is at most 0.20 in all five seeds.
- Hoeffding-certified recursive harness evolution also achieves nonzero
  coverage and positive population net gain in 4/5 seeds; seed 13 retains the
  zero-action base incumbent.
- The unchanged empirical-Bernstein certificate has zero coverage in 5/5
  seeds, confirming that the gain in power comes from the alternative valid
  concentration bound rather than a changed target.
- Binary selective risk can exceed 0.20 under a continuous-shortfall
  certificate. These are now treated as distinct estimands throughout the
  method and results.

# 2026-08-21 — H9 train-only generic surrogate pilot

- Replaced hidden-outcome-centered noisy observations with frozen train-only
  character-ngram ridge tools and passed an explicit invariance audit after
  corrupting every audited non-training label and hidden margin.
- Seed-7 strict risk-controlled coverage fell to zero for every method. This is
  a valid negative result: the earlier semi-synthetic tool quality does not
  transfer automatically to a train-only learned surrogate.
- Diagnostic worthiness AUCs were only 0.58 for unique discovery, 0.53 for
  extreme properties, and 0.64 for Matbench pairwise. The verification model
  was worse than screening for extreme properties.
- The failure localizes to scientific tool quality rather than calibration or
  leakage. H9.1 therefore replaces generic text regression with task-structured
  composition, crystal, and molecular features while retaining train-only
  fitting and the same outcome-invariance audit.

# 2026-08-21 — H9.1 structured-surrogate pilot

- Structured composition/crystal/SMILES tools passed the hidden-outcome
  invariance audit and substantially improved held-out tool ranking.
- Verification worthiness AUC improved from 0.58 to 0.93 on unique discovery
  and from 0.64 to 0.90 on Matbench pairwise. Extreme properties remained weak
  at 0.46 AUC.
- The primary sparse SD-Worth method still had zero strict coverage. Full
  verification obtained a nonzero acceptance certificate but negative held-out
  selected gain after paying all-action tool cost.
- The result shows that the scientific tool signal exists but is diluted by the
  generic gain-head layer and is not visible to the base-score acquisition
  policy. H9.2 tests direct tool-margin-to-gain calibration and deterministic
  zero-cost scientific descriptors at the base stage.

# 2026-08-21 — H9.2 direct tool-to-gain pilot

- Added deterministic scientific descriptors to the base observation and
  calibrated the structured tool-margin mean plus base-internal and
  tool-external variances directly to strict net gain.
- The seed-7 primary method still had zero exact-risk coverage and negative net
  gain from acquisition cost, so the preregistered pilot gate failed and no
  five-seed confirmatory run was launched.
- Together H9–H9.2 show that leakage-free learned proxies are insufficient for
  a publishable runtime claim even when their aggregate AUC is high. The next
  experiment must use genuine domain tools: composition featurizers/property
  models for inorganic materials and RDKit property computation for molecules.

# 2026-08-21 — H10 real-scientific-tool pilot

- Installed and versioned Matminer/Pymatgen/RDKit and replaced generated test
  observations with recomputable domain-tool outputs.
- Invalid generated SMILES are treated as tool parse failures with high
  external uncertainty; hidden generated properties are never used as a
  fallback.
- Seed 7 passes the preregistered H10 pilot gate for
  `task_sparse_tool_margin_continuous_sd_worth`: 1.51% coverage, 8.51% binary
  selective risk, +0.0552 selected mean gain, and +0.000454 population net
  gain after normalized verification cost.
- Selected actions in this pilot come from Matbench pairwise. Five-seed
  confirmation and task coverage are required before a broad multi-task claim.

# 2026-08-21 — H10 five-seed falsification

- All five runs passed the non-training-outcome corruption audit, proving that
  Matminer/RDKit visible outputs do not depend on test outcomes.
- The H10 fixed primary method passes only seed 7: one of five seeds has
  nonzero coverage and positive net gain. Mean population net gain is
  -0.000135 because the policy pays verification cost even in seeds where the
  final exact certificate accepts no action.
- Source and no-source direct tool calibrators are numerically identical at the
  decision level in this experiment; H10 does not establish a variance-
  decomposition advantage.
- The next hypothesis is certificate-aware acquisition: jointly select a
  finite acquisition policy and execution threshold on the independent
  acceptance split with multiplicity correction, and choose the no-op policy
  whenever no certified policy has positive acceptance net gain.

# 2026-08-21 — H11 joint-policy pilot

- Jointly correcting five acquisition fractions and eight execution coverages
  reduced `delta` enough that no acquisition policy retained nonzero certified
  coverage on seed 7.
- The harness correctly selected no-op and avoided cost, but failed the pilot
  requirement to preserve H10's positive seed-7 gain.
- H11.1 removes acceptance-time acquisition search: one feedback-fitted
  acquisition contract is frozen, and acceptance only activates that contract
  if it is certified and has positive acceptance net gain; otherwise it chooses
  the trivially safe no-op contract.

# 2026-08-21 — H11.1 frozen-contract activation pilot

- Seed 7 activates the frozen task-sparse real-tool contract: acceptance
  coverage is 2.02%, exact risk UCB is 0.1971 at alpha 0.20, and empirical
  acceptance population net gain is +0.000871.
- The frozen threshold transfers to held-out test with 1.51% coverage, 8.51%
  selective risk, and +0.000454 population net gain after tool cost, exactly
  preserving the positive H10 pilot.
- Source-decomposed and no-source variants remain decision-identical. This
  protocol establishes neither a source-decomposition advantage nor multi-task
  coverage; selected actions remain Matbench pairwise only.
- The activation criterion uses the sign of empirical acceptance gain. It is
  risk-certified but not gain-certified. The next rigorous contract must pair
  the exact selective-risk UCB with a simultaneous lower confidence bound on
  population net gain.

# 2026-08-22 — H12 benchmark-validity repair

- Auditing H10 revealed that the five nominal seeds reuse identical held-out
  regimes for Matbench pairwise and unique discovery. They are repeated source
  partitions, not five independent regime-generalization tests.
- The legacy reservation is the 90th percentile of outcome-conditioned base
  gain. A small change from 10.04% to 9.88% training positives makes the
  unique-discovery reservation jump from approximately 0.825 to zero.
- Added deterministic five-fold group coverage: all 28 pairwise, 7 unique, and
  10 extreme-property groups appear in exactly one test fold with no overlap.
- Added the train-only `potential_success_slot_value` reservation, based on
  normalized utility minus action cost before multiplication by the success
  label. Its across-fold range is below 2.3e-5 for every task.
- H12 keeps the real-tool method fixed and treats this as a mandatory validity
  repair. Results from the legacy repeated-regime seed protocol remain useful
  diagnostics but are no longer sufficient evidence for held-out-regime
  generalization.

# 2026-08-22 — H11.2 failed-seed certificate diagnostic

- Recomputed legacy real-tool seed 1 with full acceptance-grid diagnostics.
- The top-2% prefix has 3 errors among 41 actions (7.32% empirical risk), but
  its multiplicity-corrected exact UCB is 0.2372, narrowly above alpha 0.20.
- At the same error rate, roughly 82 selected acceptance examples would produce
  an exact UCB near 0.176. Thus acceptance power contributes to zero coverage.
- Ranking quality collapses after the top 2%: the 5% prefix has 42.16%
  empirical risk. More calibration data alone cannot create broad coverage;
  stronger scientific ranking is still necessary.

# 2026-08-22 — local MatXBot runtime audit

- Located `/Users/huan/matxbot`, whose orchestrator already exposes explicit
  scientific stages and an auditable run/state contract.
- OpenClaw hooks and the `session-logs` skill are disabled, and the gateway log
  is not a structured pre-action/outcome trajectory source.
- The clean integration point is each MatXBot stage transition: emit a
  pre-action event before an expensive transition and append tool/lab outcomes
  afterward, keeping hidden outcomes out of the decision-time event.
- Real MatXBot trajectory count remains zero until structured logging is
  enabled and actual runs are executed.

# 2026-08-22 — H12 true-regime fold-0 falsification

- The stable-reservation and true-group-fold audits pass, as does the hidden
  outcome corruption audit for all three tasks.
- The primary real-tool harness obtains zero certified coverage and activates
  no-op; the preregistered fold-0 gate fails, so folds 1--4 are not run under
  H12.
- The top-1% acceptance prefix has zero errors among 19 actions, but its exact
  family-wise UCB is 0.2344. The top-2% prefix has 2/37 errors and UCB 0.2213.
  Both narrowly miss alpha 0.20 due to acceptance-set power.
- The 5% prefix has 29.67% empirical risk, so ranking quality also limits
  broader coverage. More acceptance data is justified only for recovering the
  narrow high-score region.
- Fold 0 is now development evidence and cannot be reused as confirmatory.
  Any acceptance-allocation change must be frozen before evaluating untouched
  folds 1--4.

# 2026-08-22 — H13 powered-acceptance development pass

- Changed only the source allocation to train/feedback/acceptance =
  0.50/0.15/0.20 while preserving the H12 test fold, target, model, tools,
  alpha, multiplicity correction, and cost.
- The primary selects 36/3528 acceptance actions with 2.78% empirical risk and
  exact UCB 0.1831, then transfers to fold-0 test with 0.51% coverage, 3.33%
  risk, +0.0706 selected mean gain, and +0.000228 population net gain.
- Base reliability, ensemble LCB, verbal confidence, and evidence heuristic
  all have zero exact-risk-controlled coverage under the identical records.
- The no-source ablation also passes and is only 8.17e-6 lower in population
  net gain. Source decomposition remains an unproven contribution.
- The development gate passes. The implementation and protocol are now frozen
  for untouched confirmatory folds 1--4; fold 0 is excluded from inference.

# 2026-08-22 — H13 powered-acceptance confirmatory result

- Fold 0 was development-only. The frozen contract was then evaluated without
  tuning on untouched true-regime folds 1--4.
- All four hidden-outcome corruption audits passed. The exact selective-risk
  target of 0.20 was respected on every fold; fold 1 safely chose no-op, while
  folds 2--4 activated with test risks 0.0588, 0.1149, and 0.0833.
- Positive population net gain occurred in three of four folds. Mean coverage
  was 0.011613 and mean population net gain after acquisition cost was
  +0.000426, compared with -0.000138 for base mean and ensemble LCB and zero
  for verbal-confidence and evidence-heuristic baselines.
- The locked confirmatory gate nevertheless failed. Every selected test action
  came from Matbench pairwise, so multi-task scientific discovery was not
  established. The mean source-decomposition advantage over the no-source
  calibrator was only +0.000007 and occurred in one of four folds.
- H13 therefore establishes a narrow, risk-controlled positive-gain region and
  a static-baseline advantage, but not the claimed internal/external
  decomposition advantage or broad multi-task action worthiness. Folds 1--4
  are now consumed and may not be reused as untouched confirmation for H14.

# 2026-08-22 — H14 source calibration and frozen-policy development

- Added task/source monotonic calibration from native physical margins into a
  common gain space, followed by reliability-only source fusion. The learned
  high-fidelity weights are 0.965 for pairwise and extreme and approximately
  1.0 for unique discovery, resolving the earlier physically incomparable
  source scales.
- Replaced acceptance-time prefix search with a feedback-frozen method and
  numeric threshold followed by one independent exact acceptance certificate.
  The protocol change is documented in Amendment 3 and was made before any H14
  method was run on external folds 1--4.
- On fold-0 development, the all-source extreme-property policy selected 20
  acceptance actions with zero errors; its exact risk UCB is 0.1851 at
  alpha=0.20. The unchanged test threshold selects 28 zero-error actions with
  positive population gain. Pairwise and unique remain underpowered or
  unstable, so this is not confirmatory multi-task evidence.

# 2026-08-22 — H14 two-stage tail-risk value-of-evidence pilot

- Implemented cross-fitted base and post-tool gain posteriors, a cross-fitted
  counterfactual VoE ensemble, and a feedback-only joint acquisition/base
  execution/tool execution contract. Acceptance evaluates the frozen contract
  once and test copies it unchanged.
- The source-aware expected-VoE policy is certified on external extreme
  discovery: acceptance acquires 36 records, executes 20, has 0 errors and an
  exact risk UCB of 0.1851. Test acquires 48, executes 25 with 0 errors, and
  obtains +0.003072 population net gain after all high-fidelity tool costs.
- The matched no-external-variance VoE ablation is not certified on extreme:
  it executes only 7 acceptance actions and has risk UCB 0.4428. Raw variance
  heuristics also fail. This is the first development-only result where source
  decomposition changes a deployability decision and improves net gain.
- Pairwise fails acceptance despite positive low-risk fold-0 test behavior,
  indicating threshold-transfer instability. Unique discovery remains no-op.
  A balanced classifier for rare positive VoE does not repair either task, so
  model capacity is not the next bottleneck.
- The next method change is constrained to train-only scientific transforms
  that map each tool output directly into estimated utility/gain units before
  fusion. Further classifier or threshold tuning on fold 0 is disallowed.

# 2026-08-22 — H15 powered unique and H17 pairwise evidence-value results

- H15 repaired the underpowered unique-material benchmark using 10,987
  Matbench log-KVRH materials and chemical-system-disjoint splits. On fold 0,
  source-aware expected VoE passed the independent exact certificate with 179
  acceptance executions, zero errors, UCB 0.0166, and +0.003788 test population
  gain. The matched no-source method reached +0.003404 and the static external
  variance rule +0.003684.
- H16 preserved the original near-neighbor pairwise construction and failed:
  every acquisition method selected no-op. Diagnostics showed that the visible
  sources had AUROC approximately 0.46--0.52 for positive tool value, while the
  high-fidelity score itself was nearly perfect. The acquisition problem was
  statistically unidentifiable, so H16 remains a negative result.
- H17 introduced a hidden-outcome-free train-only mixture of easy, ambiguous,
  and OOD pair states and added a numeric agent-prior source. Fold-0 development
  selected a regularized linear source-aware expected-VoE posterior, which
  certified and obtained +0.000912 test gain versus +0.000882 for matched
  no-source and +0.000832 for static external variance.
- The H17 protocol was locked before folds 1--4. The preregistered expected-VoE
  primary failed confirmation: it certified 2/4 folds and averaged +0.000552,
  below matched no-source expected VoE at +0.000615. This claim must not be made.
- The predeclared auxiliary source-aware positive-VoE probability score
  certified 3/4 folds and averaged +0.001207 deployed test gain, 35.5% above
  matched no-source probability (+0.000890); static external variance certified
  0/4 folds. This is encouraging secondary evidence, not a replacement for the
  failed primary endpoint.
- Fold 3 exposed base-execution regime shift. The acceptance certificate worked
  as intended and prevented unsafe deployment, but the policy needs a more
  conservative verify-before-execute contract before another external test.

# 2026-08-22 — live MatXBot trajectory instrumentation

- Added a MatXBot pre-action/post-outcome event schema, atomic JSONL logger, and
  orchestrator logging policy under `/Users/huan/matxbot`.
- The event contract emits only numeric uncertainty outputs and explicitly
  forbids `selected_route` and hidden outcomes in pre-action events.
- Added `capture_mode` so live runtime, offline replay, and synthetic smoke data
  cannot be conflated.
- Added a harness-side importer that assembles one JSON record per complete
  trajectory while retaining every action-level step and linked outcome.
- End-to-end schema/import tests pass. There are still zero live MatXBot
  trajectories because no gateway or historical runtime session is available
  on this machine; instrumentation readiness is not counted as real evidence.

# 2026-08-22 — verify-before-execute contract diagnosis

- Reused consumed H17 folds 0--4 strictly as development diagnostics and marked
  them non-confirmatory.
- Disabling unverified base execution removed the fold-3 regime-shift failure.
  Source-aware expected VoE certified 5/5 folds with mean deployed test gain
  +0.001309; source-aware positive-VoE probability certified 5/5 with +0.001284.
- Matched no-source expected and probability scores also certified 5/5 with
  +0.001283 and +0.001277. Static external variance certified only 1/5 and
  averaged +0.000233.
- The contract change strongly improves safety and stability versus static
  reliability, but source decomposition adds only a small mean improvement on
  this reused dataset. A new external property dataset is required before this
  can support a paper claim.

# 2026-08-22 — H18 external dielectric confirmation

- Built a new powered pairwise benchmark from Matbench dielectric, distinct from
  the log-KVRH development data. Pair assignment uses only train-fit surrogate,
  source-disagreement, and OOD signals; chemical-system groups are disjoint and
  hidden dielectric outcomes are not used for stratification.
- Fold 0 showed that one pair per material was underpowered for the exact
  certificate. The powered protocol froze three pairs per material before
  confirmatory folds, yielding 6,723 train, 2,019 feedback, 2,691 acceptance,
  and about 2,859 test actions per fold.
- The verify-before-execute source-aware probability policy certified all four
  untouched folds and obtained mean deployed test gain +0.001209. Static
  external variance certified 0/4 and obtained zero deployed gain.
- The preregistered source-decomposition primary nevertheless failed because
  matched no-source probability also certified 4/4 and achieved +0.001212. The
  mean difference was -0.00000326. H18 therefore confirms the value of learned,
  cost-aware VoE plus verification and risk certification, but not an aggregate
  gain advantage from source decomposition.
- The secondary source-aware expected-VoE score averaged +0.001192 versus
  +0.001172 for matched no-source expected VoE, but this endpoint was not the
  preregistered primary and the difference is small.

# 2026-08-22 — H18 GPT direct-judge pilot

- Updated the direct-judge contract to return only numeric `p_success`; removed
  the route field from both prompt and structured-output schema.
- The configured endpoint exposed GPT-5.6 Sol, Luna, and Terra variants but no
  Claude models. GPT-5.6 Sol and Luna each scored the same deterministic 100
  H18 fold-1 test actions with zero API failures.
- Under a fixed top-10 acquisition budget and the same frozen high-fidelity
  execution threshold, source-aware positive-VoE scored +0.001716 population net
  gain. GPT-5.6 Luna scored -0.001160 and GPT-5.6 Sol -0.001273; static external
  variance scored -0.002000.
- GPT judges had substantially better ordinary action-label Brier scores, but
  positive-VoE AP was only about 0.186 versus 0.645 for the source-aware VoE
  score. This supports the central distinction between confidence that an
  action succeeds and whether buying evidence is worth its cost.
- This is a 100-action ranking/cost pilot, not yet a full independently
  risk-certified LLM baseline. Claude remains unavailable without another
  endpoint or Anthropic credential.
# 2026-08-22: H19 frozen transfer failure and H20 multi-tool validation

- Froze H19 phonon pairwise validation before construction (`1404b7c7163b7f8480c16a330e4de352ab638ce59bb9bdce2416f554c7cf4ef6`).
- H19 primary `verify-before-execute + source_voi_expected` certified on 3/5 folds with mean deployed gain `+0.001406`; matched no-source reached `+0.001470`; static external variance reached `+0.000490`. H19 failed its frozen success gate.
- Diagnosed two distinct failures: absolute score thresholds can shift across group-held-out folds, while source disagreement alone does not identify reducible evidence value.
- A rank-budget contract avoided threshold collapse but reached only 4/5 H17 certificates and remained below its no-source ablation; it was rejected.
- Replaced the previous two-term variance summary with an orthogonal balanced two-factor decomposition: internal replica main effect, external source main effect, and replica-by-source interaction. The components sum exactly to total variance.
- The stricter decomposition remained close to no-source on consumed H17 folds, so variance decomposition is retained as a calibrated state representation rather than claimed as a standalone performance gain.
- Implemented numeric per-tool VoE and a familywise risk-certified tool-subset evolution gate. The estimator emits `VoE(composition)`, `VoE(structure)`, and `VoE(high_fidelity)`; it never emits a route label.
- Naive Bonferroni harmed the safety baseline. A frozen weighted gatekeeping rule now reserves 90% of delta for the high-fidelity safety contract and 10% for expansion subsets. On consumed H17 folds this preserved 5/5 deployment and source-aware gain `+0.001356` versus no-source `+0.001304`, but selected high-fidelity on all folds.
- Froze H20 `matbench_perovskites` before local parsing (`6e29eb36b0065749fcdd0c7f908875750a5a7a56470557bbe9b92e9cd7f91c03`). Snapshot SHA-256 is `4641e2417f8ec8b50096d2230864468dfa08278dc9d257c327f65d0305278483`.
- H20 completed all five chemical-system-disjoint folds over the 18,928-row
  perovskite snapshot. The source-aware multi-tool policy certified 5/5 folds
  with mean test gain `+0.001390`, while the matched no-source policy reached
  `+0.001410`; static reliability certified 0/5 and deployed at zero gain.
- H20 failed its frozen primary gate because source-aware VoE did not beat
  no-source or fixed high-fidelity, and no fold selected a cheaper tool. The
  defensible result is safe fallback plus a static-baseline advantage, not
  successful cost-sensitive multi-tool routing.

# 2026-08-22 — Live OpenClaw/MatBot runtime capture and historical replay

- Installed an isolated OpenClaw 2026.5.18 runtime over the real high-energy-
  density-battery MatBot workspace without modifying the source workspace.
- Captured one pre-action runtime turn from GPT-5.6 Sol and two independent
  turns from GPT-5.6 Luna. The agents made 59 real workspace tool calls in
  total, including 11 recorded tool errors, and each proposed a concrete
  electrolyte experiment with a numeric uncertainty/action-worthiness record.
- Imported all three sessions into the versioned MatBot trajectory schema with
  `capture_mode=live_runtime`, preserved tool traces, and confirmed that no
  hidden outcome or route label entered the decision-time record.
- The two Luna replicas exposed a self-report failure: reported internal
  uncertainty changed only `0.01` (`0.43` to `0.42`) and reported VoE was
  identical (`0.94`), yet action token Jaccard was only `0.1791` and the derived
  pre-outcome net-gain proxy changed by `0.2698`. Internal uncertainty must be
  estimated from repeated action/value distributions, not accepted as a verbal
  scalar.
- All three live trajectories remain scientifically incomplete because no post-execution
  outcome is available. Their self-reported internal/external uncertainty is
  therefore uncalibrated and cannot yet support a real-world benefit claim.
- Found a real MatBot LFP history database with 12 actually measured A/B
  formulation pairs across three protocols. Converted it into 12 complete
  offline-replay trajectories: the pre-action event exposes formulation and
  protocol only, and the terminal post-outcome event restores measured DCR and
  cycle-derived utility.
- The historical replay is explicitly development-only: the source was
  inspected before protocol construction and is too small for confirmation.
  It proves the real-outcome trajectory path and supplies examples for future
  harness replay, but it is not evidence for the primary performance claim.

# 2026-08-22 — Scalar boundary-instability ablation rejected

- Added a decision-boundary instability term equal to binary vote entropy times
  squared gain range for internal replicas and external sources. Nonnegative
  weights were learned inside the heteroscedastic hurdle calibrator.
- On five consumed H17 folds, D3 marginally improved mean Brier and ECE but
  worsened AURC and log loss on every fold. In the full fold-0 VoE pipeline,
  source expected-VoE test gain decreased from `+0.001017` to `+0.000992`.
- The scalar term is therefore rejected as the primary method. It measures
  whether scores cross zero for one fixed action, not whether the scientific
  agent proposes materially different actions under repeated runs.
- Converted the 12 real LFP experiments into 19 outcome-blind A/B action-choice
  instances. Stable record-ID hashing determines orientation without outcomes;
  both candidates' measured utilities remain hidden until evaluation. This is
  the new development substrate for empirical action-distribution uncertainty.

# 2026-08-22 — Real LFP action-distribution negative result

- Completed the frozen 19-record historical LFP action-choice study with GPT-5.6
  Sol, Luna, and Terra, three replicas each: 171/171 model cells and no missing
  responses.
- Raw multi-model probability achieved Brier `0.19017`, log loss `0.56970`, and
  AURC `0.30734`. The preregistered action-entropy correction degraded these to
  `0.27136`, `0.73694`, and `0.57091`; self-report shrinkage failed similarly.
- Across records, empirical action entropy had only `0.0117` correlation with
  direct-probability error. Choice instability is real but is not a monotone
  estimate of error probability.
- Rejected direct entropy shrinkage. Implemented an exact vector-valued ANOVA
  decomposition of crossed internal-replica by external-evidence action
  distributions, plus a one-sided normalized split-conformal lower bound for
  net scientific gain.
- The conformal bound provides marginal gain coverage under exchangeability;
  conditional selective risk remains separately certified on an independent
  acceptance set. No route label is emitted.

# 2026-08-23 — Untouched superconductivity confirmation and LLM power extension

- Added H23 on the official 16,414-row Matminer superconductivity dataset. A
  deterministic parser rule removed eight invalid composition strings, leaving
  16,406 records; outer folds are chemical-system-disjoint and all labels,
  utility thresholds, and source models are train-only.
- Froze rank-budget `source_voi_expected` after fold-0 development and before
  building folds 1--4. On 13,125 untouched test records, the primary certified
  in 3/4 folds, executed 396 actions with zero errors, obtained a pooled 95%
  risk UCB of `0.00754`, and positive population gain `+0.000692`.
- The matched no-source policy also certified in 3/4 folds but achieved lower
  gain `+0.000580`; static external variance never certified. The frozen H23
  primary gate therefore passed, including the predeclared source-ablation
  comparison.
- Fixed-sequence harness evolution deployed in 3/4 folds, executed 297 actions
  with zero errors, and achieved gain `+0.000605`. It retained source features
  in one fold and safely pruned them in two, supporting adaptive signal-contract
  selection rather than a claim that every source feature helps every task.
- Completed a same-evidence 485-record comparison with GPT-5.6 Sol, Luna, and
  Terra at base, screening, and verification stages. Frozen portfolio KG
  achieved gain `32.2719`, versus `28.0575` for the highest-gain LLM method and
  `26.7116` for the strongest risk-certified LLM method. The numerical method
  was itself certified at 95% UCB `0.0932`; the highest-gain LLM was not.
- A benchmark-stratified paired bootstrap on 485 records did not exclude zero
  and the one-sided sign-flip test was approximately `p=0.19` against the
  highest-gain LLM (`p=0.10` against the certified LLM). The effect is positive
  but not yet statistically significant. Before any additional LLM calls, a
  full 2,538-record power-extension protocol froze both comparators and the
  paired testing procedure.
- Completed the frozen full screening-stage comparator with 2,538 predictions
  from each of GPT-5.6 Sol, Luna, and Terra. The strongest LLM ensemble selected
  254 actions, made 25 errors, achieved empirical risk `0.0984`, CP UCB
  `0.1348`, and strict trajectory gain `153.4501`.
- On the identical records and 10% action budget, frozen portfolio KG selected
  254 actions with 2 errors and gain `195.7886`. The paired gain advantage is
  `+42.3385`; a benchmark-stratified 100,000-draw bootstrap gives 95% interval
  `[22.1953, 62.5509]`, and the one-sided sign-flip p-value is `0.00002`.
- The full H5 acceptance gate now passes every frozen check: positive gain,
  stronger than all non-LLM baselines and the strongest same-evidence LLM,
  lower selective risk, CP UCB below `0.10`, and outcome-blind Hoeffding UCB
  `0.0847` below `0.10`.
- Completed the frozen verification-stage three-model ensemble on the same
  2,538 records. It selected 254 actions with 10 errors, empirical risk
  `0.0394`, CP UCB `0.0659`, and gain `141.0945`. Although this LLM comparator
  is risk-certified, EVL exceeds it by `+54.6940`; the paired bootstrap interval
  is `[31.0198, 78.5772]` and one-sided sign-flip `p=0.00001`.
