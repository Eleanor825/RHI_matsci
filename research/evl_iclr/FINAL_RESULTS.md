# EVL-Harness: Verified Results

Updated August 23, 2026. Values below are copied from frozen JSON artifacts;
development results are not mixed into confirmatory claims.

## Main same-evidence result

All methods see the same 2,538 acceptance trajectories and use the same 10%
action budget.

| method | selected | errors | empirical risk | CP UCB | strict trajectory gain | status |
|---|---:|---:|---:|---:|---:|---|
| EVL portfolio KG | 254 | 2 | 0.79% | 2.46% | 195.789 | certified |
| factorized base, no extra evidence | 254 | 6 | 2.36% | 4.61% | 188.792 | certified |
| fixed full evidence | 254 | 3 | 1.18% | 3.02% | 168.398 | certified |
| GPT-5.6 Sol/Luna/Terra screening ensemble | 254 | 25 | 9.84% | 13.48% | 153.450 | not certified |
| GPT-5.6 Sol/Luna/Terra verification ensemble | 254 | 10 | 3.94% | 6.59% | 141.095 | certified |
| static reliability | 254 | 157 | 61.81% | 66.90% | 36.061 | not certified |

The outcome-blind Hoeffding UCB for EVL's average selected-batch risk is 8.47%
at 95% confidence, below the frozen `alpha=10%` limit. This guarantee assumes
independent bounded losses conditional on the visible batch and a harness
frozen before outcomes are observed.

## Paired significance

| comparison | gain difference | stratified bootstrap 95% interval | one-sided sign-flip p | conclusion |
|---|---:|---:|---:|---|
| EVL minus strongest frozen LLM | +42.338 | [22.195, 62.551] | 0.00002 | significant |
| EVL minus risk-certified LLM | +54.694 | [31.020, 78.577] | 0.00001 | significant |
| EVL minus fixed full evidence | +27.391 | [10.626, 43.931] | 0.00065 | significant |
| EVL minus factorized base | +6.997 | [-10.296, 24.393] | 0.217 | not significant |
| EVL mean-KG minus conservative KG-LCB | +6.383 | [-6.509, 19.307] | 0.164 | not significant |

Thus the full numerical harness decisively beats direct LLM judging and static
reliability. Portfolio evidence valuation prevents the loss caused by always
buying evidence, but its advantage over buying no evidence is positive and not
yet significant.

## Untouched extreme-property confirmation

H23 uses 16,406 valid records from the official Matminer superconductivity
dataset. Fold 0 is development; folds 1--4 are chemical-system-disjoint and
untouched before the protocol was frozen.

| method | certified/deployed folds | executed | errors | pooled risk UCB | population gain |
|---|---:|---:|---:|---:|---:|
| source-aware expected VoI | 3/4 | 396 | 0 | 0.75% CP / 6.15% Hoeffding | +0.000692 |
| matched no-source expected VoI | 3/4 | 297 | 0 | 1.00% CP | +0.000580 |
| fixed-sequence evolved harness | 3/4 | 297 | 0 | 1.00% CP / 7.10% Hoeffding | +0.000605 |
| static external variance | 0/4 | 0 | 0 | -- | 0.000000 |

Both the frozen primary and recursive-evolution gates pass. Evolution retains
source features in one deployed fold and prunes them in two, supporting
conditional signal-contract evolution rather than universal source benefit.

## Fresh scientific-tool trajectories

On 811 fingerprint-disjoint trajectories produced with real RDKit/pymatgen
tool execution:

| method | selected | errors | CP risk UCB | strict trajectory gain |
|---|---:|---:|---:|---:|
| learned EVL harness | 82 | 0 | 3.59% | 63.427 |
| strongest hand-composed real-tool score | -- | -- | -- | 59.694 |
| base numerical uncertainty | -- | -- | -- | 4.837 |
| static reliability | -- | -- | -- | -12.192 |

Across five fitting/calibration seeds, learned EVL gain is `63.795 +/- 0.226`
with zero selected errors for every seed.

## Required limitations

- The global H5 budget selects 252/254 actions from the pairwise task; it is not
  evidence of uniform gains across all three scientific tasks.
- The task-balanced H5 stress test fails the 10% risk target.
- Unique-material H21 has positive pooled gain and beats static/no-source, but
  only 2/4 folds certify under its frozen per-fold gate.
- Live MatBot/OpenClaw traces have real tool use but no post-execution outcomes;
  the 12 measured LFP outcome replays are real but small and development-only.
- The second-provider LLM replication is unavailable on the current router.
- Both frozen three-model LLM comparators are complete; a second-provider
  replication remains unavailable on the current router.
