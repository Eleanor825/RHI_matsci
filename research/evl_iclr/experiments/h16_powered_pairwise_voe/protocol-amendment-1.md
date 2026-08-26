# H16 Protocol Amendment 1: Policy-Conditioned Evidence Value

Date: 2026-08-22

## Development observation

The first H16 fold-0 run used the counterfactual target

`gain * (1[tool_score > 0] - 1[base_score > 0]) - tool_cost`.

It selected no acquisition policy even though the held-out high-fidelity tool
score ranked worthy actions almost perfectly. The failure was traced to a
contract mismatch: the evidence-value target assumed fixed zero execution
thresholds, while the deployed two-stage policy selected nonzero base and tool
execution thresholds on feedback data.

## Frozen correction

For each feedback cross-fitting fold, execution thresholds are selected using
only the complementary feedback folds. Each held-out feedback action receives
the target

`gain * (1[tool_score >= tau_tool,-fold] - 1[base_score >= tau_base,-fold]) - tool_cost`.

The provisional thresholds maximize population net gain subject to empirical
selective risk at most 0.20 over the same predeclared coverage grid used by the
final policy. The held-out action's label, gain, and score do not participate in
its provisional threshold selection.

The acquisition model is then cross-fitted on these policy-conditioned targets.
The final acquisition and execution thresholds remain selected on feedback,
certified once on acceptance, and copied unchanged to test. Acceptance and test
outcomes are never used for target construction, model fitting, threshold
selection, or method revision.

## Interpretation

This changes the learned quantity from zero-threshold decision-flip value to
the counterfactual value of evidence under the execution contract that will
actually be deployed. It does not alter the net-gain definition, tool cost,
risk level, confidence level, data splits, source views, or test protocol.

Fold 0 remains development-only. This amendment must be frozen before running
untouched confirmatory folds.
