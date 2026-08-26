# Frozen protocol: real LFP multi-task action worthiness

Frozen on 2026-08-22 before model fitting or result inspection.

## Dataset

- Source: the real LFP formulation-performance database with SHA-256
  `606f929d5ab4e563ff779d95ae72cda5ccd47779321d7acadaec194dffc227c4`.
- Thirty complete A/B formulation experiments yield 145 candidate actions and
  1,007 outcome-grounded pairwise action trajectories.
- The primary benchmark uses the 20 AEA formulations because all have the same
  protocol family and complete outcomes for five objectives: conductivity,
  formation gas, first-cycle efficiency, specific capacity, and internal
  resistance.
- Objective directions are fixed from measurement semantics. No record is
  selected by outcome value.

## Held-out regime

- Four outer folds, five candidate formulations per fold.
- Per fold, candidate identities are partitioned into six train, three
  feedback, six acceptance, and five test candidates. Every pairwise record is
  retained only when both candidates belong to the same split; all cross-split
  edges are discarded.
- Train, feedback, acceptance, and test therefore have zero candidate identity
  overlap. The final test set evaluates chemical transfer rather than
  interpolation over previously observed candidate nodes.
- Acceptance records are deterministically divided by record-ID SHA-256 parity:
  even parity fits the one-sided conformal radius; odd parity is used only for
  the exact selective-risk certificate. Neither subset tunes the predictor.

## Scientific target

The base agent proposes the candidate with larger predicted utility. Realized
comparative net scientific gain is the signed normalized utility difference
between the proposed candidate and the available alternative, minus any
evidence-acquisition cost. The common execution cost cancels because exactly one
of the two experiments is funded under the fixed budget.

The frozen normalized harness costs are `0.000` for the static predictor,
`0.002` for a full-view bootstrap ensemble, and `0.005` for the crossed
four-view harness. These are one tenth of the smallest utility-rank increment
(`1/20 = 0.05`) or less. Results are additionally reported at zero cost and at
costs `0.01` and `0.02` as sensitivity analyses; the primary cost is `0.005`.

The harness emits only numeric quantities: expected comparative gain,
`P(gain>0)`, internal variance, external variance, interaction variance,
conformal lower gain bound, and certified selective-risk threshold. It does not
emit a route label.

## Methods

1. Static full-feature utility predictor.
2. Bootstrap ensemble without external decomposition.
3. Direct LLM judge on the candidate-disjoint test actions.
4. Crossed internal/external gain distribution: bootstrap replicas crossed with
   full-composition, solvent/salt, additive/interphase, and historical-analog
   evidence views.
5. Calibrated gain distribution plus one-sided split conformal lower bound and
   independent exact selective-risk certification.

The utility predictors are objective-specific. The hurdle calibration map is
shared across the five rank-normalized objectives because some individual
feedback subsets contain fewer than two failed proposals, making a separate
positive/negative hurdle statistically unidentified. This pooling rule is fixed
before any fold produces evaluation output.

The final crossed mean is not a uniform source average. Convex nonnegative
source weights are learned exclusively from candidate-excluded out-of-fold
predictions on the train candidates: for each training edge, both endpoint
candidates are removed before producing its source matrix. The fuser has no
affine source corrections and a fixed `0.05` shrinkage penalty toward uniform
weights. Feedback fits only gain calibration; acceptance and test never fit
source reliability.

## Validity correction before final runs

An implementation sanity run revealed unconstrained ridge extrapolations outside
the mathematically possible comparative-gain support `[-1,1]` (for example,
`8.25`). Those runs are invalid and discarded. All source predictions are now
projected onto the known target support before proposal selection, variance
decomposition, or calibration. This is a support constraint implied by the
rank-normalized utility definition, not a threshold tuned on outcomes. Final
reported runs use the continuous gain calibrator followed by distribution-free
conformal correction; the earlier hurdle sanity outputs are not reported.

## Primary metrics and gate

- Population comparative net gain after evidence cost.
- Brier and log loss for positive comparative gain.
- AURC and risk-coverage curve.
- Certified coverage at `alpha=0.10`, `delta=0.05`.
- The deployed contract is fixed before acceptance: execute only when the
  one-sided conformal lower gain bound is positive. Acceptance performs one
  exact Clopper--Pearson test of this fixed contract at `delta=0.05`; no
  threshold or coverage grid is searched on acceptance.
- Candidate-cluster bootstrap confidence intervals across folds and objectives.

The primary method must improve mean certified test net gain over static
reliability and direct LLM judge, retain a valid acceptance-set certificate, and
show no material degradation in log loss. All four outer folds are reported;
failed certificates deploy zero actions rather than being omitted.
