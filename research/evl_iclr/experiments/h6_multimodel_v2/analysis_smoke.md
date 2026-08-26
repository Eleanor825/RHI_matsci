# H6.0 Smoke Analysis

The 9-action smoke run completed all `81/81` model-by-evidence cells after one
invalid numeric response was retried. Mean internal variance was `0.000902`,
mean external variance was `0.006135`, and internal variance correlated with
absolute gain error at `0.7815`.

However, all nine deterministically selected actions were unworthy. Therefore
the run validates API compatibility, complete replica construction, and
non-degenerate source variance only. Its Brier score is a negative-class
descriptive statistic; ECE and especially AURC cannot support a discrimination
claim. The partial 30-action cache remains unanalyzed because it is not a
balanced complete matrix.

