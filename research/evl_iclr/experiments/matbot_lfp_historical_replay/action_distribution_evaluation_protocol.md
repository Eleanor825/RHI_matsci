# Action-Distribution Harness Evaluation Protocol

## Status

Frozen before inspecting the full 19-instance, three-model, three-replica result.
This remains development-only because the underlying LFP source was previously
inspected and contains only three measurement batches.

## Inputs

- 19 outcome-blind within-protocol A/B choice instances;
- GPT-5.6 Sol, Luna, and Terra;
- three independent calls per model and instance;
- hidden real measurement-derived label indicating whether candidate A had
  higher scientific utility than candidate B.

## Signals

- `p_raw`: mean model-reported probability that A is better;
- `H_action`: normalized entropy of the nine empirical A/B choices;
- `U_self`: mean `1 - assessment_confidence`;
- `V_model`: variance of model-level mean probabilities.

No hidden outcome enters any signal.

## Compared Probability Estimators

1. `llm_direct_probability`: `p_raw` without training.
2. `self_report_shrinkage`: leave-one-batch-out logistic calibration with
   features `logit(p_raw)` and `logit(p_raw) * U_self`.
3. `action_distribution_harness`: the same calibration with
   `logit(p_raw)` and `logit(p_raw) * H_action`.
4. `action_distribution_plus_model`: adds
   `logit(p_raw) * sqrt(V_model)`.

The interaction form preserves the direction of the underlying preference and
allows uncertainty only to expand or shrink confidence.

## Evaluation

Each source batch is held out once. Calibrators fit on the other two batches;
all reported predictions are out-of-batch. The primary metric is Brier score.
Secondary metrics are log loss, ECE, AURC, majority accuracy, and fixed-coverage
selective risk.

The action-distribution hypothesis is supported only if the action-entropy
harness improves Brier over both direct probability and self-report shrinkage,
without worsening log loss. This development result can justify a future
held-out protocol but cannot itself support a paper-level effectiveness claim.
