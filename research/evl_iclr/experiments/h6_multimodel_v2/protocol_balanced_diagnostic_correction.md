# H6.1 Correction: Balance Budget-Conditioned Worthiness, Not Success

Date locked: 2026-08-21

The first H6.1 implementation stratified actions by the historical raw success
label before constructing budget-conditioned gain targets. Because success and
worthiness differ, the resulting evaluation labels were 4 negative and 2
positive rather than one of each per task. That run is invalid for the stated
balanced diagnostic and is retained only as an implementation audit.

The corrected diagnostic first fits train-only utility normalization and
reservation values, computes evaluation-only `worthy = 1[G > 0]`, and then
selects one worthy and one unworthy held-out action per task. All other H6.1
settings remain unchanged. The corrected run must report exactly 3 positive
and 3 negative worthiness labels before any discrimination metric is read.

