# H3.1 Protocol — Abstention Outside Option

Status: LOCKED BEFORE EXECUTION  
Locked date: 2026-08-21

## Audit Finding

H3 v1's only positive seed was invalidated by a discrete-quantile degeneracy. On seed 13, the discover-unique training success rate fell below the 10% action budget. The empirical 90th percentile therefore equaled the common failed-action base gain `-0.2`. Since the fifth percentile was also `-0.2`, the shortfall scale collapsed and failed actions received zero utility-shortfall loss.

H3 v1 remains archived as a negative implementation audit and is not evidence for the method.

## Correct Reservation

A scientific agent can decline to spend an action slot. Therefore the opportunity-cost reservation cannot be below the zero-gain outside option:

```text
reservation_task(B) = max(
    0,
    quantile_(1-B)(base_gain on train task)
)
```

All other H3 definitions, models, coverage candidates, alphas, and evaluation splits remain unchanged. The shortfall scale is recomputed from the corrected reservation.

## Success Criterion

The H3 primary criterion must hold on at least four of five seeds, not merely in the mean: non-zero coverage, positive budget-conditioned net gain, and improvement over static reliability at `alpha=0.20`.
