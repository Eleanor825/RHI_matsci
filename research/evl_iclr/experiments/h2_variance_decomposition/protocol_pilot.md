# H2 Pilot Protocol — Balanced Replica Mechanism Check

Status: LOCKED BEFORE EXECUTION  
Locked date: 2026-08-21

## Scope

This pilot validates the replica data contract and source-decomposition mechanism. It is not evidence for closed/open LLM uncertainty and does not satisfy the real-MatBot requirement.

## Replica Design

- Agent axis: three gain heads trained on train-only bootstrap replicas with seeds `{101, 202, 303}`.
- Evidence axis: the observed evidence, a label-blind leave-one-evidence-out view, and a label-blind conflicting-source challenge.
- Every audited action must have all nine cells. Missing or duplicate cells invalidate the action.
- Models are fit only on the historical train split. Hidden outcomes are not used to create evidence views.

## Decomposition

For predicted gain matrix `g[m,e]`:

```text
U_internal = mean_e Var_m(g[m,e])
U_external = Var_e mean_m(g[m,e])
U_total = U_internal + U_external
```

## Mechanism Checks

1. Conflict and evidence-dropout views should shift mean predicted gain relative to the full view.
2. Observed external variance should exceed a negative control that independently permutes evidence-view assignments within each agent row.
3. Source statistics are reported by task; no task may be silently dropped.

## Interpretation

Passing this pilot only proves that the software and operational definition distinguish aligned evidence interventions from agent bootstrap variation. The final H2 experiment still requires actual model replicas and retrieval/tool views.
