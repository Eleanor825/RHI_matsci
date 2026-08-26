# Frozen confirmatory protocol: safe crossed evidence-residual harness

Frozen on **Saturday, August 22, 2026**, before the three-task held-out result is inspected.

## Research question

Does a calibrated action-worthiness distribution that crosses internal multi-model beliefs with external scientific tool evidence improve held-out scientific-action selection over (i) a direct LLM judge, (ii) a static success-reliability gate, and (iii) individual tool-only scores, while preserving an independently certified selective-risk constraint?

## Data and role separation

- Dataset: `research/evl_iclr/data/sequential_tool_trajectories_v3_crossfit_train_900.jsonl`.
- Tasks: pairwise high-property selection, unique-material discovery, and extreme-property discovery.
- The `train` tool observations are rebuilt by five-fold group-excluded cross-fitting. No training record is scored by a tool environment fit on its own scientific group.
- `train`: group-held-out source-contract/hyperparameter selection, then final model fitting.
- `feedback`: deterministically split by scientific group into two disjoint subsets:
  - probability subset: temperature calibration only;
  - conformal subset: one-sided normalized split-conformal gain calibration only.
- `acceptance`: frozen threshold-family risk certification and harness acceptance only.
- `test`: opened once after source contract, blend, regularization, calibration, conformal radius, and threshold family are frozen.

## Numeric output contract

For action `a` in state `h`, the harness returns no route. It returns:

1. calibrated positive-net-gain probability `P(Delta U > 0 | h, a)`;
2. expected net scientific gain `E[Delta U | h, a]`;
3. internal variance;
4. external variance;
5. internal-external interaction variance;
6. irreducible outcome variance;
7. total variance;
8. a one-sided conformal lower bound on net gain.

The exact net-gain target is the benchmark-frozen `budget_conditioned_gain`, constructed from train-normalized scientific utility minus action/tool cost and failure harm. Hidden outcome fields are unavailable to every visible trajectory stage and every LLM prompt.

## Method selection

Candidate external source contracts are:

- screening only;
- verification only;
- screening plus verification.

Candidate safe blend weights are `0, 0.25, 0.5, 0.75, 1.0`; candidate L2 regularizers are `0.1, 1.0, 10.0`. Selection uses only train-group-held-out binary log loss. Brier score and expected-gain MAE are deterministic tie-breakers. Blend zero is the protected direct-LLM fallback.

The final probability grid is temperature-scaled cell-wise, then re-decomposed. Therefore the reported calibrated probability remains exactly equal to the weighted grid mean and total epistemic variance remains exactly the sum of internal, external, and interaction components.

## Baselines

- `direct_llm`: mean calibrated action-worthiness from `gpt-5.6-sol`, `gpt-5.6-luna`, and `gpt-5.6-terra`, using base-stage evidence only.
- `static_reliability`: logistic gate trained only to predict historical action success from fixed pre-action confidence/reliability features; scalar temperature mapping is fit on the probability-feedback subset.
- `static_screening_tool`: calibrated screening-stage tool positive probability.
- `static_verification_tool`: calibrated verification-stage tool positive probability.
- `ours_action_worthiness`: calibrated crossed evidence-residual positive-net-gain probability.
- `ours_conformal_lower_gain`: one-sided lower confidence score derived from expected gain and total decomposed variance.

## Risk guarantee

On the independent acceptance split, each method audits a frozen coverage grid:

`1%, 2%, 5%, 10%, 15%, 20%, 30%, 50%, 75%, 100%`.

For each coverage prefix, selected-action error is `1[Delta U <= 0]`. A Bonferroni-corrected exact Clopper-Pearson upper bound controls family-wise confidence at `delta=0.05`. The largest certified prefix is deployed unchanged on test. We report `alpha in {0.10, 0.20, 0.30, 0.40}` because the existing 98-record acceptance split cannot statistically certify useful 10% risk coverage unless the selected zero-error prefix is sufficiently large. No unsupported low-risk claim will be made.

## Primary endpoints

1. Test Brier score and log loss for calibrated action worthiness.
2. Test AURC and coverage at fixed selective risk.
3. Test expected-gain MAE/correlation.
4. Certified test coverage, selective risk, selected mean gain, and population net gain.
5. One-sided conformal gain coverage.
6. Mean internal/external/interaction variance and their association with squared gain error.

## Acceptance rule for the harness revision

The candidate crossed-evidence harness is accepted only if it obtains a non-empty independent acceptance certificate at the stated alpha and its held-out deployment does not violate the certified risk level. Predictive improvements without a certificate are reported as development evidence but do not count as successful RHI adoption.
