# H11.2 Failed-Seed Risk-Grid Diagnostic

Primary method: `task_sparse_tool_margin_continuous_sd_worth`, legacy seed 1,
`alpha=0.20`, eight predeclared coverage candidates, family-wise `delta=0.05`.

| coverage | selected | errors | empirical risk | exact UCB | certified |
|---:|---:|---:|---:|---:|---|
| 0.0054 | 11 | 0 | 0.0000 | 0.3696 | no |
| 0.0103 | 21 | 1 | 0.0476 | 0.2956 | no |
| 0.0202 | 41 | 3 | 0.0732 | 0.2372 | no |
| 0.0502 | 102 | 43 | 0.4216 | 0.5498 | no |
| 0.1004 | 204 | 142 | 0.6961 | 0.7738 | no |

The top-2% prefix has useful empirical ranking quality but misses the exact
risk target because the multiplicity-corrected upper bound is 0.2372. At the
same 7.3% empirical risk, approximately 82 selected acceptance examples would
reduce the bound to about 0.176. Larger prefixes are genuine ranking failures,
not merely low-power certificates.

The failure therefore has two causes: insufficient acceptance power at very
small coverage and rapid score degradation beyond the top 2%. Increasing
acceptance data may recover a narrow certified region but cannot replace
better scientific ranking for useful coverage.
