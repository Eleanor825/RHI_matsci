# Calibrated Action-Gain Distribution for Scientific Agents

> **Status:** current replacement method. The scalar decision-instability and
> direct entropy-shrinkage variants are retained only as rejected development
> ablations.

## 1. Estimand: net scientific gain

For visible state `x`, candidate scientific action `a`, and post-action outcome
`y`, define gain relative to the best feasible outside option as

\[
G(a;x,y)=U_{\mathrm{sci}}(a,y)-\lambda_c C(a)
-\lambda_h H(a,y)-\rho_b(x).
\]

`U_sci` is an outcome-grounded scientific utility, `C` is execution and tool
cost, `H` is failure or irreversibility harm, and `rho_b` is a reservation value
fit from training data for budget regime `b`. The estimator never receives the
hidden outcome at inference time. Its primary numeric outputs are

\[
\widehat{\mathbb E}[G\mid x,a],\quad
\widehat P(G>0\mid x,a),\quad
\widehat{\mathrm{LCB}}_{1-\alpha}(G),
\]

plus decomposed uncertainty components. It does not emit a route label.

## 2. Crossed internal/external interventions

For every action, evaluate a balanced grid with internal replicas `r=1,...,R`
and outcome-blind external evidence views `e=1,...,E`. A cell emits a numeric
action distribution `z_{r,e}` over a fixed action menu and an estimated gain
distribution. Internal replicas vary model, seed, or decoding while keeping
visible evidence fixed. External views vary only evidence available before the
action, including deterministic tool descriptors, retrieval packets,
provenance, or controlled evidence ablations.

For categorical action distributions, define the grand, replica, and evidence
means `z`, `z_r`, and `z_e`. The squared Euclidean population variance has the
exact vector ANOVA decomposition

\[
V_{\mathrm{total}} = V_I+V_E+V_{I\times E},
\]

\[
V_I=\mathbb E_r\|z_r-z\|_2^2,\qquad
V_E=\mathbb E_e\|z_e-z\|_2^2,
\]

\[
V_{I\times E}=\mathbb E_{r,e}
\|z_{r,e}-z_r-z_e+z\|_2^2.
\]

This is a property of the observed decision distribution, not a verbal
confidence proxy. It distinguishes model/trajectory instability, evidence
sensitivity, and non-additive model-by-evidence reactions.

## 3. Calibrated gain distribution

For comparative scientific actions, the method has an exact value-factorized
form. Let `P` be an internal replica's probability that the proposed candidate
is better and let `M` be an outcome-blind external evidence estimate of the
candidate utility gap. With `X=2P-1`, the crossed net gain is `G=XM-C`. Its
variance decomposes exactly as

\[
\operatorname{Var}(G)=
\mathbb E[M]^2\operatorname{Var}(X)
+\mathbb E[X]^2\operatorname{Var}(M)
+\operatorname{Var}(X)\operatorname{Var}(M).
\]

The three terms are internal decision uncertainty, external scientific-value
uncertainty, and interaction. This factorization formalizes why confidence alone
cannot answer whether an action is worth executing: the same success
probability can imply very different gain when the scientific consequence
changes.

A heteroscedastic hurdle model maps the raw gain mean and decomposed variances
to positive-gain probability, positive-gain magnitude, and negative-gain
magnitude. Variance components enter with learned nonnegative coefficients.
This yields calibrated action worthiness, expected net gain, and lower-tail
loss while preserving the distinction between probability of success and
value conditional on success.

The distributional model is trained only on the training and feedback splits.
Model selection and harness evolution cannot inspect acceptance or test
outcomes.

## 4. Distribution-free lower gain bound

On a separate calibration split, define the normalized one-sided
nonconformity score

\[
s_i=\frac{\widehat\mu_i-G_i}{\max(\widehat\sigma_i,\epsilon)}.
\]

The conformal radius is the `ceil((n+1)(1-alpha))`-th order statistic. For a
new action,

\[
L_\alpha(x,a)=\widehat\mu(x,a)-q_{1-\alpha}\widehat\sigma(x,a)
\]

has finite-sample marginal lower-bound coverage at least `1-alpha` under
exchangeability. This guarantee does not assume Gaussian residuals.

Marginal conformal coverage is not presented as a conditional selective-risk
guarantee. A frozen policy ranked by `L_alpha`, `P(G>0)`, or expected gain must
additionally pass an exact binomial upper confidence bound on an independent
acceptance set. Among certified policies, selection maximizes held-out net
scientific gain subject to the risk bound.

## 5. Self-evolving harness

RHI is used only to propose changes to the measurement contract: evidence
extractors, replica design, calibration features, and tool-value estimators.
Every proposed harness is retrained from scratch on the same development data.
It is accepted only when it improves the predeclared risk-gain objective on an
independent acceptance set and all familywise safety certificates pass. The
base scientific agent cannot rewrite labels, outcomes, utility, data splits,
or acceptance tests.

## 6. Falsifiable claims

1. Internal resampling primarily changes `V_I`; controlled evidence
   interventions primarily change `V_E`; non-additive effects appear in
   `V_IxE`.
2. Decomposed distributional uncertainty improves held-out gain calibration or
   certified net gain over matched mean-only and static-reliability models.
3. Conformal lower bounds attain nominal held-out coverage; exact acceptance-set
   certificates control conditional selective risk.
4. The complete method beats direct LLM judges, self-reported confidence,
   ensemble disagreement, static reliability, and a no-decomposition ablation
   under identical action and tool budgets.
5. Improvements transfer across materials tasks, model families, and held-out
   scientific regimes, including real MatBot and tool trajectories.

## 7. Development evidence and rejected shortcuts

The direct entropy-shrinkage shortcut failed on the 19-record historical LFP
replay. Its leave-one-batch-out Brier score was `0.27136` and AURC was `0.57091`,
versus `0.19017` and `0.30734` for the raw multi-model direct probability.
Action entropy had error correlation `0.0117`. Therefore action-choice entropy
is not used as a monotone probability correction.

This negative result motivates the crossed intervention design: observed action
variation is retained as a decomposed state variable whose usefulness must be
learned and validated, rather than assumed to imply lower confidence.
