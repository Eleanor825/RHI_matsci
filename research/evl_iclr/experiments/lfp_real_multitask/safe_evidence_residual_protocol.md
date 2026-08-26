# Frozen development protocol: Safe Crossed Evidence-Residual Harness

Frozen on 2026-08-22 before the train-action LLM grid and end-to-end results
were complete. This is a development experiment because the four LFP test
folds were inspected during the failed product-gain analysis. It cannot serve
as the final untouched confirmation experiment.

## Estimand

For a pairwise scientific action, the realized comparative net gain is the
normalized utility difference of the LLM-selected candidate relative to the
alternative, minus a fixed `0.01` method overhead. The numeric harness outputs
`P(G>0)`, expected net gain, a conformal lower gain bound, and decomposed gain
variance. It does not output a route label.

## Mean worthiness model

The three GPT-5.6 variants define the direct probability anchor. Scientific
sources are reoriented to the LLM consensus proposal before feature extraction;
their signs are never discarded. A regularized residual logistic model uses
proposal-aligned source means, source support, source dispersion, individual
source channels, and cross-model dispersion. The final probability is a convex
safe blend with the direct anchor. Hyperparameters are fixed to blend weight
`0.5` and regularization `1.0` from the earlier leave-one-candidate-fold-out
development diagnostic; no additional tuning is allowed in this run.

## Crossed uncertainty distribution

Each internal model probability is crossed with every bootstrap scientific
source cell. Counterfactual cell probabilities are moment-matched so their
weighted mean equals the calibrated action-worthiness probability. Weighted
two-factor ANOVA then gives exact internal, external, and interaction variance.
Positive- and negative-gain magnitude heads convert the probability grid into
an expected-gain distribution; the three ANOVA terms scale exactly under this
affine hurdle transformation.

## Data use

- `train`: fit the residual and magnitude heads. Every train source prediction
  excludes both endpoint candidates before fitting the scientific model.
- `feedback`: fit one-sided normalized split conformal only.
- `acceptance`: evaluate the fixed probability thresholds `0.50:0.05:0.90`
  with Bonferroni-adjusted exact Clopper-Pearson upper bounds.
- `test`: deploy only an acceptance-certified threshold.

The primary risk levels are `0.10` and `0.20`, each at familywise confidence
`1-delta=0.95`. Failure to certify deploys zero test actions.

## Development success criteria

Relative to the same three-model direct anchor: lower mean Brier score, lower
mean log loss, lower mean AURC, valid conformal coverage, and nonnegative
certified population net gain. These results guide the method freeze for a new
held-out benchmark but are not themselves a final paper claim.
