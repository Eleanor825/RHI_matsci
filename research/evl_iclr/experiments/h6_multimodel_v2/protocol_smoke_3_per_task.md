# H6.0 Addendum: Balanced Three-Per-Task Smoke Matrix

Date locked: 2026-08-21

The preregistered 30-action H6 pilot reached only 80/270 cached cells because
the provider's xhigh request throughput was too low for one research cycle.
The partial matrix is retained and is not analyzed.

Before inspecting any H6 metrics, run a smaller balanced smoke matrix using
the first three deterministic held-out actions per task under the same sample
ordering, models, prompt, and three external replicas:

```text
9 actions * 3 GPT-5.6 models * 3 verification views = 81 cells
```

The smoke result is descriptive only. It can validate API compatibility,
matrix completeness, and non-degenerate source components, but it cannot
replace the locked 30-action pilot or satisfy the final multi-model claim.

