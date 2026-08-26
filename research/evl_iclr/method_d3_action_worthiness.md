# D3: Decision-Distribution Decomposition for Scientific Action Worthiness

> **Development status:** the scalar boundary-instability version below was
> implemented and rejected as the primary method. It remains a documented
> negative ablation. The replacement method estimates the distribution of
> actual agent action choices/proposals rather than sign flips of a fixed
> action-value score.

## Scientific Decision Target

For a proposed scientific action `a` in visible state `x`, define realized net
scientific gain relative to the best available outside option as

\[
G(a;x,y)=U_{\mathrm{sci}}(a,y)-\lambda_c C(a)
-\lambda_h H(a,y)-\rho(x),
\]

where `U_sci` is outcome-dependent scientific utility, `C` is normalized action
cost, `H` is failure or irreversibility harm, and `rho` is the train-only
reservation value induced by the experimental budget. An action is worthy only
when `G>0`; confidence that an answer is correct is not the target.

## Replica-by-Evidence Decision Distribution

The harness evaluates a balanced grid of internal replicas `r=1,...,R` and
controlled external evidence states `s=1,...,S`:

\[
\widehat G_{r,s}=f_{\theta_r}(x,E_s).
\]

The grid is decomposed exactly into replica, evidence, and interaction effects:

\[
V_{\mathrm{total}}=V_{\mathrm{int}}+V_{\mathrm{ext}}+V_{\mathrm{int}\times\mathrm{ext}}.
\]

This ANOVA decomposition distinguishes model/trajectory instability from
evidence sensitivity and from non-additive model-by-evidence reactions.

## Decision-Boundary Instability

Ordinary variance does not distinguish harmless numerical dispersion from a
distribution that flips the execute/abstain decision. For marginal replica
values `m_r` and marginal evidence values `m_s`, D3 adds

\[
B_I = H_2\!\left(\Pr_r[m_r>0]\right)
      (\max_r m_r-\min_r m_r)^2,
\]

\[
B_E = H_2\!\left(\Pr_s[m_s>0]\right)
      (\max_s m_s-\min_s m_s)^2,
\]

where `H_2` is normalized binary entropy. These terms are zero when all
replicas/sources imply the same decision, even if their values differ, and grow
when scientifically meaningful action-worthiness decisions flip.

## Calibrated Action Worthiness

D3 fits a heteroscedastic hurdle distribution on feedback data. Its variance is

\[
\sigma^2(x)=\sigma_0^2+w_I V_I+w_E V_E+w_XV_X
             +\eta_I B_I+\eta_E B_E,
\]

with nonnegative learned weights. The model separately estimates probability
of positive gain and positive/negative gain magnitudes, then reports:

- calibrated `P(G>0)`;
- expected net scientific gain;
- lower-tail expected shortfall;
- a tail-risk action-worthiness score.

No route label is emitted by the estimator. A downstream contract compares
these numeric quantities with cost, risk, and evidence value.

## Selective-Risk Contract

Candidate contracts are selected on feedback only. A frozen contract is
deployed only if an independent acceptance set satisfies an exact finite-sample
upper confidence bound

\[
\Pr\left(R_{\mathrm{selective}}\le \alpha\right)\ge 1-\delta.
\]

Among certified contracts, the optimization target is population net
scientific gain after all evidence and execution costs, not raw accuracy or
coverage alone.

## Falsifiable Claims

1. D3 must improve held-out calibration or certified net gain over variance-only
   source decomposition; otherwise the boundary term is rejected.
2. D3 must outperform self-reported confidence/uncertainty, static reliability,
   and direct LLM judges under the same action budget and risk certificate.
3. Internal and external components must have distinct intervention responses:
   internal resampling should primarily change `V_I/B_I`, while controlled
   evidence changes should primarily change `V_E/B_E` or interaction.
4. Real MatBot trajectories must preserve the pre-action/outcome boundary and
   confirm that lower D3 uncertainty predicts more stable and more valuable
   executed actions.

Development experiments may choose whether D3 enters a future protocol, but no
consumed fold can be used as confirmatory evidence for these claims.

## Development Result

Across five consumed H17 folds, adding scalar boundary instability to the
variance-only calibrator changed mean test Brier by `-0.000021` and ECE by
`-0.000642`, but worsened AURC on 5/5 folds (`+0.000162`) and log loss on 5/5
folds (`+0.000908`). In the full fold-0 verify-before-execute pipeline, primary
test net gain fell from `+0.001017` to `+0.000992`.

This falsifies the claim that sign entropy of fixed-action gain replicas is a
sufficient internal uncertainty signal. The live Luna trajectories explain the
failure: nearly identical scalar uncertainty can coexist with qualitatively
different proposed experiments. The next method therefore models empirical
action distributions directly.
