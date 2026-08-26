# H20 Frozen Perovskite Multi-Tool External Validation

## Lock and independence

- Frozen before downloading, parsing, constructing, or evaluating `matbench_perovskites` locally.
- No target distribution, fold outcome, source-model performance, or harness result from H20 may change this protocol.
- The method is selected only from consumed H17 development folds after the failed H19 transfer diagnosis.
- All H20 folds 0 through 4 are confirmatory.

## Scientific task

- Source: official Matbench `matbench_perovskites`, 18,928 crystal structures.
- Original target: formation energy of the five-atom perovskite cell.
- Scientific score: negative formation energy, so larger score means a more favorable low-formation-energy candidate.
- Action: choose candidate A rather than candidate B for scientific follow-up.
- Hidden outcome: the true pair ordering and normalized utility are unavailable to the agent and all pre-acquisition estimators.

## Data regime

- Five chemical-system-disjoint outer folds.
- Within each fold, non-test chemical systems are partitioned into train, feedback, and acceptance with weights 0.50, 0.15, and 0.20.
- Easy, ambiguous, and OOD pair regimes use only train-fitted surrogate predictions and fixed proportions 0.40, 0.30, and 0.30.
- Pair multiplier: 1.0.
- Seed: 2001.
- Train fits property/source models; feedback fits numeric worthiness and contract candidates; acceptance performs familywise risk certification; test is untouched until deployment is decided.

## Frozen numeric uncertainty method

The estimator does not output a route label. For each action it emits three numeric quantities:

- `expected_voi::composition`;
- `expected_voi::structure`;
- `expected_voi::high_fidelity`.

Each value estimates the counterfactual change in scientific net gain caused by purchasing that evidence source, including its cost. The estimator uses:

- calibrated action worthiness;
- internal replica variance;
- external source variance;
- replica-by-source interaction variance;
- visible evidence support/conflict;
- OOD score;
- source reliability;
- perturbation stability;
- tool identity and cost.

The variance components use a balanced two-factor ANOVA decomposition and sum exactly to total predictive variance.

## Frozen contract evolution

- Candidate tool subsets are all non-empty subsets of composition, structure, and high-fidelity tools.
- Each subset receives a policy fit only on feedback data.
- The high-fidelity-only subset is the safety baseline.
- Familywise risk level: delta = 0.05.
- Weighted Bonferroni allocation: 90% of delta to the safety baseline; the remaining 10% is shared uniformly by expansion subsets.
- Selective-risk target: alpha = 0.20.
- Among acceptance-certified candidates, select the candidate with the highest feedback population net gain.
- If no candidate certifies, deploy nothing and assign deployed test gain zero.
- Policy mode: absolute calibrated thresholds.

## Frozen costs

- Composition evidence: 0.002 normalized population utility units.
- Structure evidence: 0.002.
- High-fidelity evidence: 0.020.

## Frozen baselines

1. Matched no-source multi-tool VoE, removing external/context signals while preserving internal stability, tool identity, and cost.
2. Fixed high-fidelity expected VoE under the same gain and risk protocol.
3. Static external-variance acquisition under the same alpha and delta where applicable.

## Confirmatory success criterion

The H20 primary succeeds only if all conditions hold:

1. source-aware multi-tool VoE deploys on at least four of five folds;
2. mean deployed test population gain is positive;
3. mean deployed test gain exceeds matched no-source multi-tool VoE;
4. mean deployed test gain exceeds fixed high-fidelity expected VoE;
5. at least one non-high-fidelity tool is selected by a certified contract on at least two folds;
6. every deployed fold has observed test selective risk at or below alpha.

Failure of any condition is reported without changing the primary, costs, alpha, delta, or candidate family.
