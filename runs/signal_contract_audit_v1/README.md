# Uncertainty Signal Contract Audit

This audit checks whether every record contains the complete decision-time
internal and external uncertainty contract.

The current 20,987-record historical benchmark does **not** pass the strict
contract:

- available internal signals: `verbal_confidence`, `perturbation_stability`;
- missing internal signals: `self_consistency`, `semantic_entropy`,
  `model_disagreement`, `calibrated_probability`, and all answer-logit
  difference fields;
- available external signals: `evidence_support`, `evidence_conflict`,
  `source_reliability`, `ood_score`, `source_risk`;
- missing external signal: `tool_agreement`.

The implementation now exposes a strict signal-contract mode. A full-signal
experiment must enrich the records first and enable strict mode; otherwise
training raises an error rather than silently replacing unavailable signals
with zero. Answer-logit difference features can be attached through
`AnswerLogitSignalProvider` when the evaluated model exposes next-token logits.

See `signal_coverage.json` for per-task counts.
