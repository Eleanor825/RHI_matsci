# H6 Protocol: Closed-Model Replica Pilot on Sequential Benchmark v2

Date locked: 2026-08-21

## Goal

Test whether the v2 task-native tool observations produce meaningful
model-internal and evidence-external variation for real closed-model calls.
This is a pilot for the multi-model requirement, not the final GPT/Claude/open
comparison.

## Matrix

- Tasks: all three material-discovery tasks.
- Actions: 10 deterministic held-out test actions per task.
- Internal/model replicas: `gpt-5.6-sol`, `gpt-5.6-luna`,
  `gpt-5.6-terra`.
- External replicas: three independent verification observations generated
  under the locked v2 observation process.
- Required cells: `30 * 3 * 3 = 270`; missing cells invalidate the pilot.

Every model receives the same prompt contract and no hidden outcome. It must
return numerical `p_positive_gain`, `expected_gain`, `p_success`, expected
normalized utility, and assessment confidence. No route is requested.

## Endpoints

1. Brier, ECE, and AURC of the raw model-ensemble `p_positive_gain`;
2. expected-gain MAE;
3. mean internal and external variance;
4. correlation of each variance component with absolute gain error;
5. per-task and per-model diagnostics.

## Interpretation boundary

A positive result requires a complete matrix and non-degenerate source
components. It does not establish superiority over H5.2 because the pilot has
no independent LLM calibration/acceptance split. Claude and open-model
families remain missing if the current provider does not expose them.

