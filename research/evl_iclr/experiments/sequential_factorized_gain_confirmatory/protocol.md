# Frozen H2 protocol: factorized scientific gain harness

Frozen on **Saturday, August 22, 2026**, before predictions for the 2,100 newly added trajectories are complete and before any newly added acceptance/test outcome is inspected.

## Motivation fixed from the rejected H1 revision

The earlier residual classifier improved held-out Brier score but failed its independent acceptance test. Its false positives were mostly actions that were likely to succeed but whose scientific value did not exceed the action-slot reservation value. H2 therefore replaces direct worthiness classification with the structural factorization

`net_gain = success * conditional_scientific_utility - action_cost - failure_harm - task_reservation_value`.

This is a preregistered revision motivated by the old 900-record development/acceptance data. Every record present in that dataset is excluded from H2 acceptance and test evaluation.

## Data roles

- Expanded dataset: `sequential_tool_trajectories_v3_crossfit_train_3000.jsonl`.
- Old development IDs: all records in `sequential_tool_trajectories_v3_crossfit_train_900.jsonl`.
- Train: all expanded train records; tool features are five-fold group-excluded cross-fitted.
- Feedback: all expanded feedback records, split by scientific group into probability-calibration and conformal-calibration subsets.
- Fresh acceptance: expanded acceptance records absent from the old 900-record dataset.
- Fresh test: expanded test records absent from the old 900-record dataset.
- The fresh evaluation is record-disjoint, not fully group-disjoint from the old design-development evidence. This limitation must remain explicit.

## Structured model

For each internal model replica `m` and external evidence replica `e`, H2 estimates:

1. `P(S=1 | m,e)`, action success probability;
2. `E[U | S=1,m,e]`, conditional normalized scientific utility;
3. `P(G>0 | m,e) = P(S=1 | m,e) * P(U > reservation + cost | S=1,m,e)`;
4. `E[G | m,e]` under the frozen cost and failure-harm weights.

The direct LLM worthiness probability remains a protected convex fallback. Candidate blend weights are `0, .25, .5, .75, 1`. Candidate regularizers are `.1, 1, 10`. Candidate source contracts are screening, verification, and both. Selection uses train-group-held-out worthiness log loss only; Brier score and gain MAE are deterministic tie-breakers.

## Internal and external replicas

- Internal replicas: `gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.6-terra` numerical `p_success`, expected normalized utility, and direct positive-gain probability.
- Every scientific tool stage is expanded into three declared-uncertainty evidence replicas at standardized quantiles `-1, 0, +1`, weighted `.25, .50, .25` and multiplied by source fidelity.
- The model therefore always produces a non-degenerate internal-model by external-evidence grid, even if method selection retains only one tool stage.
- Weighted crossed ANOVA exactly decomposes probability and gain variance into internal, external, and interaction components.

## Calibration and guarantee

- Temperature scaling is applied cell-wise to the worthiness grid on the probability-feedback subset.
- Expected gain receives a one-sided normalized split-conformal lower bound on the disjoint conformal-feedback subset.
- The independent fresh acceptance split audits the frozen coverage grid `1%, 2%, 5%, 10%, 15%, 20%, 30%, 50%, 75%, 100%` using Bonferroni-corrected exact Clopper-Pearson bounds.
- The **primary policy** is fixed in advance by the scientific resource constraint: select the top `10%` actions independently within each task. Because this is one pre-specified policy rather than a searched threshold family, its exact Clopper-Pearson certificate uses `delta=.05` without a multiplicity penalty. The coverage grid remains a secondary audit.
- Reported target risks are `.10, .20, .30, .40, .50`; no certificate is claimed when the exact upper bound exceeds alpha.

## Baselines and primary endpoints

Baselines are the calibrated three-model direct LLM judge and the fixed-feature static success-reliability gate. Primary endpoints are Brier score, log loss, ECE, AURC, expected-gain MAE/correlation, exact variance accounting, variance-error association, conformal lower-bound coverage, and certified selected-action net gain.

## Revision acceptance rule

H2 is accepted as the next harness revision only if the fresh acceptance set yields a non-empty exact risk certificate and the frozen policy respects that risk level on fresh test. Predictive improvement without certification is evidence for the estimator but not evidence for successful safe self-evolution.
