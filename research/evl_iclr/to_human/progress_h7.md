# SD-Worth / EVL Harness Research Progress

## Current method

- The host MatBot proposes a scientific action.
- The uncertainty harness estimates a continuous distribution over
  budget-conditioned net scientific gain.
- A complete model-replica x tool-replica matrix separates internal and
  external uncertainty.
- The scalar runtime output is calibrated action worthiness
  `q = P(G > 0 | visible pre-action information)`.
- The host sparsely acquires verification evidence using task-specific
  thresholds learned on feedback data.
- An untouched acceptance split chooses a risk-controlled execution threshold.
- H7 recursively mutates the harness contract and accepts a revision only when
  family-wise certified coverage is nonzero and acceptance net gain improves.

## Data and benchmark

- 20,987 action records across three scientific tasks.
- 45 scientific regimes with complete-group held-out test splits.
- 900 released sequential trajectories: 300 per task.
- Every trajectory contains base, screening, and verification observations,
  explicit tool cost, and provenance.
- Hidden outcomes and gain targets are separated from visible observations.
- The current v2 tool channel is semi-synthetic controlled replay, not live
  MatBot runtime.

## H5.2 fixed source-decomposed harness

- Five split seeds; exact risk target `alpha=0.20`; verification cost `0.002`.
- Nonzero risk-controlled coverage: 5/5 seeds.
- Positive population net gain: 5/5 seeds.
- Mean coverage: 1.43%.
- Mean selective risk: 4.95%.
- Mean verification acquisition: 28.0%.
- Mean population net gain: +0.001402.
- Beats the no-source continuous ablation in 3/5 seeds.
- Paired Wilcoxon p-value: 0.3125; superiority is not yet statistically
  decisive.

## H7 recursive harness evolution

- Family-wise correction covers five harness candidates and eight coverage
  thresholds.
- Nonzero test coverage: 5/5 seeds.
- Test risk at most 0.20: 5/5 seeds.
- Positive population net gain: 5/5 seeds.
- Mean population net gain: +0.003043.
- Median population net gain: +0.000538.
- Mean selective risk: 3.04%.
- Four different final harness contracts are selected across five seeds,
  showing contract-level rather than per-action evolution.

## Real GPT-5.6 replica diagnostic

- 54/54 complete cells: Sol, Luna, Terra x three verification views x six
  worthiness-balanced actions.
- Raw direct judge: Brier 0.3681, ECE 0.4919, accuracy 0.50.
- Internal variance/error correlation: 0.8667.
- External variance/error correlation: 0.6766.
- Interpretation: real-model disagreement is useful as an uncertainty input,
  but raw LLM confidence is not a reliable final worthiness score.
- Boundary: exploratory label-balanced diagnostic; natural 30-action matrix,
  Claude, and open-model families remain incomplete.

## Claims currently supported

1. Strict action worthiness can be defined as positive budget-conditioned net
   scientific gain, distinct from success.
2. Internal/model and external/evidence variation can be operationally
   decomposed with balanced replica matrices.
3. Continuous source-decomposed gain calibration restores useful exact
   risk-controlled coverage on an identifiable sequential benchmark.
4. Sparse evidence acquisition is necessary for positive cost-adjusted gain.
5. Acceptance-set recursive harness evolution can select different contracts
   across regimes while preserving family-wise exact binary risk control.

## Claims not yet supported

1. Nonzero bounded utility-shortfall-certified coverage.
2. Live MatBot/tool trajectory transfer.
3. Claude and open-model family replication.
4. Significant universal superiority over all strong static and LLM-judge
   baselines.
5. Final ICLR-level evidence on real scientific runtime outcomes.
