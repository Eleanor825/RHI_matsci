# Frozen development protocol: crossed action distributions

Frozen on 2026-08-22 before inspecting any crossed-view model outputs.

## Data boundary

- Nineteen real historical LFP pairwise action choices from three measurement
  batches.
- Every target batch is held out from the historical retrieval library.
- Target DCR, cycle, conductivity, label, and utility are absent from all
  model prompts. Outcomes of other-batch historical analogs may be retrieved.
- This is development-only because the underlying 12 experiments were already
  inspected before benchmark construction.

## Crossed design

- Internal replicas: GPT-5.6 Sol, Luna, and Terra, with two independent calls
  per model (`R=6`).
- External evidence views (`E=3`): base formulation/protocol; deterministic
  directional formulation descriptors; nearest historical analog outcomes
  retrieved only from non-target batches.
- Total planned cells: `19 * 6 * 3 = 342`.

## Operational audit before result inspection

An initial monitor incorrectly counted the four top-level cache keys instead of
the entries under `predictions`, making an actively progressing `xhigh` run look
stalled. Before any prediction values were inspected, the error was identified:
the original run had completed 232/342 cells. A temporary `medium` restart was
stopped and archived uninspected. The experiment therefore retains the original
preregistered `xhigh` reasoning effort and resumes its original cache. No
hypothesis, view, split, metric, or decision rule changed.

## Frozen diagnostics

1. Decompose both probability vectors and sampled choices into exact internal,
   external, and interaction population variance.
2. Report Brier, log loss, ECE, AURC, and threshold accuracy for every evidence
   view and their untrained mean.
3. Report correlations of every variance component with base-view binary and
   squared error. These are diagnostics, not confirmatory significance claims.

## Development decision rule

- Direct entropy or variance shrinkage remains rejected regardless of this
  result.
- Promote crossed variance into the next calibrated gain experiment only if at
  least one externally enriched view improves base-view Brier without worse log
  loss and an external or interaction component has positive squared-error
  correlation.
- Otherwise retain the decomposition as a measurement primitive and redesign
  evidence acquisition; do not tune this consumed dataset into a positive
  result.
