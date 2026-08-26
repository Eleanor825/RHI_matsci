# H1.2 Protocol — Task-Conditioned Worthiness Heads

Status: LOCKED BEFORE CONFIRMATORY EXECUTION  
Locked date: 2026-08-21

## Motivation

The H1.1 seed-7 development run established that the budget-conditioned target is distinct from success, but a pooled linear head produced zero risk-controlled coverage. A post-hoc signal audit on seed 7 showed strong task heterogeneity: visible features were substantially more informative for the pairwise and extreme-property tasks than for unique-material discovery. This motivates task-conditioned prediction rather than forcing one shared feature-to-gain map across scientifically different objectives.

## Development and Confirmation Split

- Seed `7` is development-only because its acceptance labels were inspected during diagnosis.
- Seeds `{1, 13, 21, 42}` are untouched confirmatory seeds.
- No hyperparameter or method choice may be changed after inspecting confirmatory results.

## Target

Use the locked H1.1 train-only empirical utility normalization and task-specific reservation value at budget fraction `B=0.10`.

```text
gain_i(B) = base_gain_i - reservation_task(i)(B)
worthiness_i(B) = 1[gain_i(B) > 0]
```

## Models

Compare the locked pooled models against:

1. `task_conditioned_binary_worthiness`: one binary worthiness head per scientific task, independently temperature-calibrated on the feedback split;
2. `task_conditioned_gain_distribution`: one continuous gain-distribution head per scientific task, independently calibrated on the feedback split.

All heads use the same visible feature contract, training epochs, regularization, data splits, and no oracle outcome features.

## Risk Control

Use the locked coverage grid `{0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}` and one-sided exact binomial upper bounds with Bonferroni correction across the eight candidates. The selected coverage is applied as a deterministic score-ranked prefix on test.

## Outcomes

- Primary: risk-controlled net gain per action at `alpha=0.30`.
- Secondary: coverage and selective risk at `alpha in {0.10, 0.20, 0.30}`, Brier score, ECE, AURC, and worst-regime empirical risk.
- A useful positive result requires non-zero confirmatory coverage and higher net gain per action than pooled binary worthiness and static reliability.

## Interpretation Rule

If all task-conditioned models retain zero coverage at `alpha=0.30`, H1.2 is refuted and the historical visible feature contract is considered insufficient for a strict action-worthiness guarantee. The project must then obtain richer model/tool replicate signals rather than tuning the same static features.
