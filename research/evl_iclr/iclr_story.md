# ICLR Story: Evidence-Valued Uncertainty for Scientific Agents

## One-sentence problem

Scientific agents can propose plausible actions, but existing confidence and
judge methods do not answer whether an action is worth spending scientific
resources on now rather than obtaining more evidence or not acting.

## Core tension

High confidence is not high scientific value. A likely-to-succeed action can be
scientifically trivial, while a high-value action can be uncertain but worth a
cheap simulation. More evidence can reduce uncertainty but still lower total
scientific gain after its cost is charged. The decision object must therefore be
net action value, not confidence or route classification.

## Method in one paragraph

EVL-Harness is a route-free numerical layer placed between an agent proposal
and execution. It estimates success, conditional utility, strict net gain,
internal uncertainty, external uncertainty, interaction uncertainty,
aleatoric uncertainty, calibrated positive-gain probability, and a conformal
gain lower bound. Under a shared scientific budget, evidence is acquired by its
predicted marginal effect on the selected top-k portfolio relative to an exact
leave-one-out replacement action. Recursive harness improvement changes the
signal contract, gain estimator, calibration rule, or evidence-allocation rule
by generation and accepts a revision only after a frozen held-out risk/gain
gate passes.

## Three contributions

1. **Decision target.** Formalize scientific-agent uncertainty as the
   distribution of strict marginal scientific gain rather than answer
   correctness, verbal confidence, or a route label.
2. **Numerical harness.** Introduce crossed internal/external uncertainty
   decomposition, calibrated action-worthiness, and feedback-calibrated
   portfolio knowledge gain for cost-aware evidence acquisition.
3. **Evaluation protocol.** Provide action/outcome-linked scientific
   trajectories, actual pymatgen/RDKit tool executions, fingerprint-disjoint
   held-out regimes, and explicit selected-set risk certificates against static
   reliability and same-evidence LLM judges.

## Current confirmatory evidence

### Adaptive evidence benchmark

- Fresh acceptance: 2,538 actions.
- Primary portfolio-KG harness: 254 selected, 2 errors.
- Empirical selective risk: 0.79%.
- One-sided 95% risk UCB: 2.46%.
- Strict trajectory gain: 195.79.
- Strongest completed non-LLM comparator: base-only 188.79.
- Frozen strongest same-evidence LLM comparator: three-model GPT-5.6 screening
  ensemble with gain 153.45, 25 errors, 9.84% empirical risk, and 13.48% CP
  risk UCB.
- Paired numerical-minus-LLM gain: +42.34 over the same 2,538 records; the
  benchmark-stratified bootstrap 95% interval is [22.20, 62.55] and the
  one-sided sign-flip p-value is 0.00002.
- The numerical harness has 2 errors, 0.79% empirical risk, 2.46% CP UCB, and
  an outcome-blind Hoeffding selected-batch UCB of 8.47% at alpha=10%.

### Live scientific-tool benchmark

- Development: 3,601 live-runtime trajectories and 7,202 action steps.
- Fresh fingerprint-disjoint test: 811 trajectories and 1,622 steps.
- Primary learned harness: 82 selected, 0 errors.
- One-sided 95% risk UCB: 3.59%.
- Strict trajectory gain: 63.43 versus 59.69 for the strongest real-tool
  hand-composed score, 4.84 for base numerical uncertainty, and -12.19 for
  static reliability.
- Five fitting/calibration seeds: 0 errors for every seed and strict gain
  63.79 plus/minus 0.23.

## What the paper must not claim

- It must not claim that ANOVA decomposition, conformal calibration, value of
  information, or harness evolution is individually new.
- It must not call historical benchmark outcomes new wet-lab experiments.
- It must not call the selected-set Clopper--Pearson certificate an action-wise
  distribution-free theorem.
- It must not claim uniform task gains: the shared global budget currently
  concentrates on pairwise materials decisions, and the task-balanced stress
  test remains weaker for unique and extreme discovery.

## Reviewer-facing answer to “is this just engineering?”

The components are familiar, but the optimized object and their coupling are
not a pipeline convenience. Changing the target from correctness to strict
scientific gain changes supervision, calibration, evidence valuation,
selection, and the statistical claim. The exact portfolio replacement price
also resolves a concrete inconsistency between evidence acquisition and final
execution that caused the earlier method to fail on fresh feedback.

The evidence-acquisition claim is deliberately narrower than the full-system
claim. Portfolio KG significantly beats always acquiring full evidence
(`+27.39`, paired 95% interval `[10.63, 43.93]`, one-sided `p=0.00065`), showing
that evidence must be valued rather than added indiscriminately. Its gain over
the strong no-evidence factorized ablation is positive (`+7.00`) but not
significant (`p=0.217`). The decisive LLM and static-baseline advantage should
therefore be attributed to calibrated numerical action valuation as a whole;
portfolio KG is a cost-aware evidence-allocation component, not evidence that
additional evidence helps every batch.

## Remaining acceptance-critical evidence

1. Add a second provider/model family when an endpoint is available.
2. Treat task-balanced unique/extreme results as an explicit limitation or
   improve them in a separately frozen next-generation experiment.
3. Add a larger runtime-plus-outcome MatBot study; current real-runtime and
   measured-outcome evidence exist in separate, small datasets.
