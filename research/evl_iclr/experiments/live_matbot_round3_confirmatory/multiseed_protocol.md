# Post-Confirmatory Multi-Seed Robustness Protocol

Recorded on August 23, 2026, after the frozen primary round-3 evaluation. This
analysis cannot change the primary claim or tune the test result.

The same 811-record test is evaluated with fitting/calibration seeds 7, 42,
1729, 20260823, and 314159. The architecture, features, budget, risk level, and
test records remain unchanged.

One seed produced a calibration partition with no positive extreme-property
examples. The frozen primary evaluator correctly refused to fit logistic
regression on a single class. For robustness evaluation only, a wrapper uses a
Laplace-smoothed constant probability `(positives+1)/(n+2)` when the calibration
labels contain one class; otherwise it delegates exactly to the frozen
evaluator. This is a numerical edge-case rule, not a test-tuned model change.

The primary seed 20260823 must reproduce the frozen primary result. Multi-seed
statistics are reported as post-confirmatory robustness evidence only.
