# H6.1 Exploratory Addendum: Label-Balanced API Diagnostic

Date locked: 2026-08-21

This addendum is written after discovering that H6.0's deterministic smoke
sample contains nine negative labels. It is explicitly exploratory and cannot
be used as confirmatory benchmark evidence.

For each task, select one positive and one negative held-out action using the
evaluation-only label, then run the same three GPT-5.6 models and three
independent verification views:

```text
6 actions * 3 models * 3 evidence views = 54 cells
```

The sole purpose is to check whether real-model source variances and numerical
gain estimates remain non-degenerate when both classes are represented. Label
stratification must be disclosed in every report, and the resulting Brier/ECE/
AURC cannot be compared directly with naturally sampled test performance.

