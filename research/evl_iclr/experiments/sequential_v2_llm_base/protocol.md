# Frozen sequential-v2 generic LLM baseline protocol

Frozen on 2026-08-22 before any prediction from this run is inspected.

- Dataset: `sequential_tool_trajectories_v2_900.jsonl`.
- Split: the existing frozen `test` split only (`109` trajectories expected).
- Evidence: base-stage pre-action observation only.
- Models: `gpt-5.6-sol`, `gpt-5.6-luna`, and `gpt-5.6-terra`.
- Reasoning effort: `medium`; one call per model/action.
- Output: numeric `p_positive_gain`, expected gain, success probability,
  normalized utility, and assessment confidence. No route is requested.
- Budget conditioning: train-derived task reservation values, cost weight `0.15`,
  and failure-harm weight `0.25`.
- Hidden worthiness, utility, and gain are excluded from every EvidenceView and
  are opened only after all `109 x 3` cells are complete.

Primary metrics are Brier score, log loss, AURC, ECE, accuracy, expected-gain
MAE, and the association between cross-model variance and gain error. This is
the frozen direct-judge baseline for the later safe evidence-residual harness.
