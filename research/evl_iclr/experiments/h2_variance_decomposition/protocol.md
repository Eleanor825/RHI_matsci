# H2 Protocol — Operational Internal/External Variance Decomposition

Status: DESIGN LOCKED; EXECUTION REQUIRES REPLICATES  
Locked date: 2026-08-21

## Hypothesis

For a fixed scientific action, variation across agent replicas with evidence fixed predicts agent-caused failures, while variation across evidence/tool replicas after averaging agent replicas predicts evidence-caused failures. Their combination improves worthiness calibration under prompt and evidence shifts.

## Operational Decomposition

For predicted gain `g_hat[m,e]` from agent replica `m` and evidence view `e`:

```text
U_internal = mean_e Var_m(g_hat[m,e])
U_external = Var_e mean_m(g_hat[m,e])
U_total    = U_internal + U_external
```

Agent replicas may be open-model seeds/checkpoints or repeated black-box samples. Evidence views must come from auditable retrieval/tool bootstraps, not hidden labels.

## Required Data

- At least three agent replicas per action.
- At least three evidence/tool views per action.
- The complete balanced prediction matrix for each audited action.
- Hidden outcomes used only after predictions are frozen.

## Confirmatory Prediction

`U_internal` rises under prompt/model perturbation while `U_external` rises under evidence removal/conflict/OOD interventions. A shuffled source assignment should destroy this specificity.
