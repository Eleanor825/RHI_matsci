# Frozen direct-LLM baseline protocol

Frozen on 2026-08-22 before any of the 200-record outputs are inspected.

- Records: union of the four candidate-disjoint outer test sets, exactly 200
  real outcome-grounded pairwise actions across five LFP objectives.
- Models: GPT-5.6 Sol, Luna, and Terra.
- Inference: one independent structured-output call per model and action, for
  600 planned calls.
- The prompt exposes formulation, objective, protocol family, and pre-action
  evidence only. Hidden measurements, normalized utility, labels, and outcome
  differences are absent.
- Primary direct baseline is the equally weighted mean `p(A better)` across the
  three models. Model-specific results are also reported.
- No model, action, or prompt variant is selected after seeing outcomes.
- This test set has already been used for deterministic method development, so
  results are external-development evidence rather than final confirmation.

## Operational amendment before output inspection

The initial all-record `xhigh` run completed only 27/600 calls in approximately
19 minutes. No prediction values were inspected; its cache is archived as
`run_xhigh_uninspected_incomplete/`. The complete 600-call comparison uses
`medium` reasoning effort for all three models. A separate outcome-blind
50-record subset will later be evaluated at `xhigh` to check reasoning-effort
sensitivity. No record, model, prompt, or metric changed.
