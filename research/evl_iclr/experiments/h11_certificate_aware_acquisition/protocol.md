# H11 Protocol: Certificate-Aware Evidence Acquisition

Date locked: 2026-08-21

## Problem

H10 acquires real scientific-tool evidence using a feedback-fitted base-score
threshold before the final execution certificate is known. In four seeds the
execution certificate selects zero actions, but the harness still pays tool
cost for 8–20% of actions.

## Joint finite policy selection

For the frozen H10 direct source-decomposed score, predeclare global acquisition
fractions `{0.02, 0.05, 0.10, 0.20, 0.30}`. On the independent acceptance
split, each fraction defines a base-score top-prefix acquisition policy. For
each policy, apply the unchanged eight-point exact-risk execution-coverage
grid. Split `delta=0.05` across acquisition policies; the existing selector
then splits each policy's delta across execution coverages.

For every simultaneously certified pair, compute acceptance population net
gain after charging `0.002` for every acquired action. Select the certified
pair with highest positive acceptance net gain. If none is positive, select the
no-op contract, acquire no tool evidence, and execute no action. Freeze both
thresholds before test.

The method still outputs only scalar action worthiness after evidence is
acquired. The host uses the frozen acquisition threshold to decide whether the
scalar requires the expensive scientific tool.

## Pilot gate

Run seed 7 first on H10 real scientific tools. Proceed to five seeds if test
risk is at most 0.20 and population net gain is non-negative, while preserving
or improving H10 seed-7 positive gain. Five-seed success requires non-negative
population net gain in all seeds, positive gain in at least four, and nonzero
coverage in at least four.
