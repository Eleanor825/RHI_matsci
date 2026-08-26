# H2.1 Protocol — Closed-Model × Retrieval-Tool Replica Pilot

Status: LOCKED BEFORE EXECUTION  
Locked date: 2026-08-21

## Purpose

Measure operational internal/model and external/tool uncertainty using real model calls and auditable historical retrieval tools. This 30-action pilot validates signal semantics and cost before a full confirmatory run; it is not the final multi-model benchmark.

## Actions

- Use split seed `7` from the locked historical benchmark.
- Select exactly 10 test actions per task with deterministic sampling seed `260712447`.
- The target action's label, utility, and gain are excluded from every prompt.
- Train-only utility normalization and reservation values use budget fraction `B=0.10`.

## Agent Replica Axis

Use identical prompts and three independently served closed models:

- `gpt-5.6-sol`
- `gpt-5.6-luna`
- `gpt-5.6-terra`

Each returns only numerical action-worthiness quantities: `p_positive_gain`, `expected_gain`, `p_success`, expected normalized utility, and assessment confidence. No route is requested or returned.

## Evidence/Tool Axis

For every target action, create three auditable views:

1. original pre-action evidence;
2. five nearest historical analogs retrieved from the train split, including only those analogs' historical outcomes;
3. task-level train-library success and utility statistics.

The target record and any matching raw record identifier are forbidden from the retrieval library results.

## Required Matrix

Every action must have all `3 models × 3 evidence views = 9` predictions. Failed or missing cells invalidate that action rather than being imputed.

## Primary Checks

1. Compare base-view ensemble Brier/ECE with all-tool-view ensemble Brier/ECE for budget-conditioned worthiness.
2. Report internal variance, external variance, and their correlation with absolute gain error.
3. Report source components by task and individual model.
4. Inspect whether retrieval views systematically change predicted gain in scientifically interpretable directions.

## Decision Rule

Proceed to a larger H2 confirmatory run only if the matrix is complete and at least one source component has a positive association with held-out gain error or errors are materially reduced by tool evidence. Otherwise revise the model output target or retrieval tools before scaling calls.
