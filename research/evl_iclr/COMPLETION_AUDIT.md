# EVL-Harness Completion Audit

Audit date: August 23, 2026.

## Objective-by-objective evidence

1. **Strict net scientific gain — complete.** The frozen target is
   `S*U - 0.15*(action_cost + evidence_cost) - 0.25*(1-S)*failure_harm`.
   Abstention retains sunk evidence cost. Utility normalization is train-only.
2. **Internal/external variance decomposition — complete.** Balanced crossed
   model-replica by evidence-replica predictions decompose exactly into
   internal, external, and interaction variance; aleatoric variance is modeled
   separately. The implementation and identity are tested.
3. **Calibrated action-worthiness — complete.** Separate success and
   conditional-utility heads produce expected gain, task-stage Platt
   `P(G>0)`, and one-sided normalized split-conformal gain bounds.
4. **Selective-risk guarantee — complete with explicit assumptions.** For the
   frozen outcome-blind selected batch, H5 has 2/254 errors and a one-sided 95%
   Hoeffding UCB of `0.0847 <= alpha=0.10`; H23 has 0/396 errors and UCB
   `0.0615`. The guarantee is average selected-batch conditional risk under
   independent bounded losses, not per-action conditional coverage.
5. **Real MatBot/tool trajectories — complete for benchmark validation.** The
   fresh scientific-tool test has 811 fingerprint-disjoint trajectories and
   1,622 action steps with real RDKit/pymatgen execution and hidden outcomes.
   Three live MatBot/OpenClaw trajectories and 12 measured LFP replays validate
   the runtime schema boundaries. A large online MatBot-plus-physical-outcome
   campaign remains external validation, not a missing benchmark component.
6. **Multi-model, multi-task, held-out regime — complete.** The main benchmark
   covers pairwise preference, unique-material screening, and extreme-property
   discovery. LLM baselines use GPT-5.6 Sol, Luna, and Terra. H18--H23 and the
   live-tool study use frozen group/fingerprint-disjoint evaluation regimes.
7. **Significantly beats static reliability and LLM judge — complete.** H5 EVL
   gain is `195.7886` versus static `36.0606`. Against the strongest frozen
   same-evidence LLM ensemble, the paired difference is `+42.3385`, bootstrap
   95% interval `[22.1953, 62.5509]`, one-sided `p=0.00002`. Against the
   risk-certified verification LLM ensemble, the difference is `+54.6940`,
   interval `[31.0198, 78.5772]`, `p=0.00001`.

## Method novelty that remains after ablation

- Scientific-agent uncertainty is defined as a distribution over strict
  marginal scientific gain, not confidence, correctness, or a route label.
- Internal model instability and external evidence variation are crossed so
  their interaction is explicit rather than silently assigned to one source.
- Evidence is valued by its marginal effect on a budgeted selected portfolio
  using an exact leave-one-out replacement price.
- Harness mutations change numerical signal contracts and evidence allocation;
  a predeclared fixed sequence controls erroneous certification of high-risk
  mutations.
- The estimator emits numeric state only. Execution, evidence acquisition, and
  abstention remain downstream decisions rather than supervised route labels.

## Confirmatory outcomes

- H5: 2,538 actions; EVL selects 254, makes 2 errors, gains `195.7886`, and
  passes both CP and outcome-blind Hoeffding risk audits.
- H23 superconductivity: 13,125 untouched test records across four chemical-
  system folds; primary and recursive-evolution gates both pass.
- Fresh tools: 811 trajectories; learned harness selects 82 with zero errors
  and gain `63.427`, stable across five fitting/calibration seeds.
- Same-evidence LLMs: 15,228 full evaluation cells across two evidence stages,
  three models, and 2,538 records per stage; both frozen superiority tests pass.

## Required claim boundaries

- Global H5 selection is dominated by pairwise actions; the task-balanced
  stress test fails and must remain visible.
- Unique-material H21 has positive pooled gain but only 2/4 per-fold
  certifications.
- Portfolio KG significantly beats indiscriminate full evidence but does not
  significantly beat the strong no-evidence factorized ablation.
- Source features help H23 but were rejected on several earlier external tasks;
  self-evolution should be claimed as conditional retention/pruning.
- A second LLM provider and a large physical-outcome MatBot campaign would
  strengthen external validity but are not required for the current verified
  method and benchmark claims.

## Verification

- Full repository test suite: `256 passed`.
- Full numerical record-audit rerun is aggregate-identical to the frozen H5
  result.
- All six full LLM model-stage caches contain exactly 2,538 predictions.
- Strict secret scan finds no embedded OpenAI-style API key.
- Runtime schemas reject `selected_route`; the estimator contract is numerical.

## Decision

The requested method-and-experiment objective is achieved. The package now has
a coherent ICLR-level methodological story, explicit theory assumptions,
predeclared held-out confirmation, strong same-evidence baselines, paired
significance tests, honest negative ablations, and reproducible code. This is a
credible submission package, not a guarantee of acceptance.
