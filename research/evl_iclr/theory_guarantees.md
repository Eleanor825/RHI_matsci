# Theory and Guarantee Scope

## 1. Strict scientific gain

For a proposed scientific action `a` under evidence state `z`, the realized
decision target is

\[
G=S U-\lambda_C(C_a+C_z)-\lambda_H(1-S)H_a.
\]

`S` is the post-action success indicator, `U` is train-normalized scientific
utility conditional on success, `C_a` and `C_z` are action and acquired-
evidence costs, and `H_a` is failure harm. Abstention after evidence acquisition
has gain `-lambda_C C_z`; evidence is therefore never free after it is used.
All normalization, weights, and action budgets are frozen before the evaluated
split.

## 2. Exact crossed decomposition

Let `g_me` be the numerical gain estimate from model replica `m` and evidence
replica `e`. Define the model main effect, evidence main effect, and residual
interaction by

\[
I_m=\bar g_{m\cdot}-\bar g,\qquad
E_e=\bar g_{\cdot e}-\bar g,\qquad
R_{me}=g_{me}-\bar g-I_m-E_e.
\]

Because each main effect and every corresponding residual margin sum to zero,
the cross terms vanish, giving the exact finite-grid identity

\[
\frac1{ME}\sum_{m,e}(g_{me}-\bar g)^2
=\frac1M\sum_m I_m^2+
\frac1E\sum_e E_e^2+
\frac1{ME}\sum_{m,e}R_{me}^2.
\]

This identifies operational internal, external, and interaction variability on
the crossed replica grid. It does not identify unique causal uncertainty
sources without interventions.

## 3. Outcome-blind selected-batch risk

Let a harness and its top-`k` scoring rule be frozen before an independent
evaluation batch. Selection may depend on all visible covariates and predicted
scores in the batch but not on outcomes. Conditional on those covariates and
the selected index set, assume selected losses

\[
L_i=\mathbb 1\{G_i\leq0\}\in[0,1]
\]

are independent; their conditional means may differ. For `m` selected actions
and empirical risk `r_hat`, Hoeffding's inequality yields

\[
\Pr\left(
\frac1m\sum_{i\in A}\mathbb E[L_i\mid X_{1:n}]
>\hat r+\sqrt{\frac{\log(1/\delta)}{2m}}
\ \middle|\ X_{1:n}
\right)\leq\delta.
\]

Top-`k` ranking does not invalidate this statement: after conditioning on the
outcome-blind scores, the selected indices are fixed. The result bounds average
conditional risk of that selected batch. It is not individual-action
conditional coverage and does not cover correlated outcomes without a cluster
extension.

At `delta=0.05`, the frozen H5 acceptance result has `2/254` errors and
Hoeffding UCB `0.0847`; H23 has `0/396` errors and UCB `0.0615`. Both satisfy
the frozen `alpha=0.10` limit. Clopper--Pearson values are reported only as a
tighter auxiliary audit under the stronger i.i.d. Bernoulli interpretation.

## 4. Risk-safe fixed-sequence evolution

Consider a predeclared ordered list of harness mutations
`H_1,...,H_J`. For each candidate, test the null that its selected-batch risk
exceeds `alpha` with a level-`delta` one-sided test, and stop at the first
non-rejection. Let `j*` be the first true null. A high-risk candidate can enter
the certified prefix only if the test falsely rejects `H_j*`; therefore

\[
\Pr(\text{any high-risk harness enters the certified prefix})\leq\delta.
\]

This fixed-sequence familywise statement allows dependent test statistics but
requires the candidate order to be frozen before inspecting the acceptance
outcomes. If mutations are generated adaptively from acceptance feedback, a
fresh acceptance set or reusable-holdout mechanism is required.

Within the certified prefix, the current implementation selects higher
acceptance-set net gain. That selection does not itself prove population gain
improvement. Consequently, gain is reported on a separate untouched test fold;
the theorem is a risk-safety statement, not a monotonic-improvement theorem.

## 5. Claim boundary

The defensible theoretical contribution is the coupling of:

1. a strict cost- and harm-aware scientific gain target;
2. an exact crossed operational uncertainty decomposition;
3. calibrated gain and evidence-value estimates;
4. outcome-blind selected-batch risk control; and
5. fixed-sequence risk-safe harness mutation.

The work does not claim universal task improvement, causal identifiability of
uncertainty components, per-action conditional coverage, or guaranteed gain
monotonicity under adaptive reuse of one acceptance set.
