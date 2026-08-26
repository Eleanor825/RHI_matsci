# Evidence-Valued Uncertainty Harness

## 1. Decision object

At scientific state `s`, the underlying agent proposes an action `a`. The
harness does not classify a route and does not replace the agent. It estimates
the distribution of strict marginal scientific gain

\[
G(s,a,z)=S(s,a,z)U(s,a,z)
-\lambda_C\{C(a)+E(z)\}
-\lambda_H[1-S(s,a,z)]H(a),
\]

where `z` is the acquired evidence state, `S` is observed success, `U` is
normalized scientific utility conditional on success, `C` is action cost, `E`
is evidence cost, and `H` is failure harm. Current experiments freeze
`lambda_C=0.15` and `lambda_H=0.25` before evaluation.

The numerical output is

\[
H(s,a,z)=\left(
\hat p_{S},\hat\mu_U,\hat\mu_G,\hat p_{G>0},
\hat V_{int},\hat V_{ext},\hat V_{int\times ext},
\hat V_{alea},\widehat{LCB}_{1-\alpha},\widehat{VoI}
\right).
\]

There is no `selected_route`. An external controller may execute when the gain
lower bound is positive, acquire evidence when its portfolio value exceeds its
cost, or abstain otherwise.

## 2. Signals

### Internal signals

Internal signals describe instability of the proposing model under changes
that do not add scientific evidence.

- stochastic response replicas;
- prompt-paraphrase and context-order replicas;
- verbal confidence and response consistency;
- action identity and utility consistency across model replicas;
- for open-weight models only: token entropy, sequence likelihood, hidden-state
  geometry, energy scores, checkpoint ensembles, and layerwise perturbations.

Closed models use only API-observable replicas, structured probabilities when
the API exposes them, and perturbation stability. Open models may add white-box
signals, but the numerical output contract is identical.

### External signals

External signals describe uncertainty in the scientific evidence and action
environment.

- source reliability and provenance risk;
- evidence support and contradiction;
- distance from the training/reference domain;
- coverage and nearest-reference distance;
- predictive uncertainty and disagreement of retrieved tools or simulations;
- missing measurements and parse/tool failures;
- action cost, evidence cost, reversibility, and failure harm.

Every declared signal is logged numerically. The learned heads receive the full
internal, external, cost, and tool feature maps; feature importance is reported
to reveal unused or redundant signals rather than claiming that every signal
must matter on every task.

## 3. Crossed variance decomposition

For `M` model replicas and `E` evidence replicas, let `g_me` be the predicted
strict gain. Define

\[
\bar g_{m\cdot}=E^{-1}\sum_e g_{me},\quad
\bar g_{\cdot e}=M^{-1}\sum_m g_{me},\quad
\bar g=(ME)^{-1}\sum_{m,e}g_{me}.
\]

The operational decomposition is

\[
V_{int}=M^{-1}\sum_m(\bar g_{m\cdot}-\bar g)^2,
\]

\[
V_{ext}=E^{-1}\sum_e(\bar g_{\cdot e}-\bar g)^2,
\]

\[
V_{int\times ext}=(ME)^{-1}\sum_{m,e}
(g_{me}-\bar g_{m\cdot}-\bar g_{\cdot e}+\bar g)^2.
\]

For the balanced crossed design,

\[
\operatorname{Var}_{m,e}(g_{me})
=V_{int}+V_{ext}+V_{int\times ext}.
\]

This is an exact ANOVA identity. It is an operational source decomposition, not
a claim that model and evidence uncertainty are uniquely causally identifiable
without interventions. Aleatoric outcome variance is estimated separately and
added only after the epistemic decomposition.

## 4. Calibrated action-worthiness

The factorized estimator has separate success and conditional-utility heads:

\[
\hat\mu_G=\hat p_S\hat\mu_U
-\lambda_C(C+E)-\lambda_H(1-\hat p_S)H.
\]

Task-stage Platt scaling calibrates `p(G>0)`. A one-sided normalized split-
conformal residual produces

\[
LCB_i=\hat\mu_{G,i}-q_{1-\alpha}\hat\sigma_i.
\]

Under exchangeability between calibration and evaluation examples, the usual
split-conformal statement is marginal coverage for a newly drawn action. It is
not an assumption-free conditional guarantee for every action or every task.

## 5. Portfolio evidence value

Execution has a shared top-`k` budget. Evidence therefore matters only when it
can change the selected portfolio. For action `i`, let `tau_-i` be the exact
leave-one-out replacement score of the best alternative portfolio. The
one-step portfolio knowledge gain is

\[
KG_i(z\rightarrow z')=
\mathbb E\left[(LCB_i(z')-\tau_{-i})_+
-(LCB_i(z)-\tau_{-i})_+\right]
-\lambda_C\Delta E_i.
\]

Current and future cells use the same conformal execution score and remove sunk
evidence cost before comparison. A task-transition bootstrap residual head is
trained on the realized change in strict top-`k` portfolio gain caused by the
evidence acquisition. The target is numerical marginal portfolio value, never
a route label.

## 6. Selective-risk statement

The executor selects at most `k=ceil(0.10 n)` actions with positive calibrated
gain lower bounds. The harness, calibration maps, evidence policy, and budget
are frozen before the independent evaluation batch, and selection is blind to
evaluation outcomes. Conditional on the visible batch and selected indices,
assume the bounded losses

\[
L_i=\mathbb 1\{G_i\leq0\}\in[0,1]
\]

are independent, although they need not have a common error probability. If
`e` of `m` selected actions fail, Hoeffding's inequality gives the one-sided
selected-batch average-risk bound

\[
\bar R_{sel}\leq \frac em+
\sqrt{\frac{\log(1/\delta)}{2m}}
\]

with probability at least `1-delta`. This remains applicable when top-`k`
ranking couples which actions are selected because, after conditioning on the
outcome-blind scores, the selected index set is fixed; it does not require the
selected Bernoulli losses to be identically distributed.

The one-sided Clopper--Pearson bound is also reported as a tighter auxiliary
audit under the stronger i.i.d. Bernoulli interpretation. Neither statement is
distribution-free conditional coverage for every individual action. Correlated
outcomes require a group- or cluster-level extension and are listed as a claim
boundary.

## 7. Harness self-improvement

The harness does not rewrite itself after every action. One generation is:

1. collect pre-action numerical signals and post-outcome strict gains;
2. propose a new signal contract, estimator, calibration rule, or evidence-
   allocation rule;
3. fit only on training and feedback data;
4. evaluate on a disjoint acceptance set;
5. accept the new harness only if the frozen risk and net-gain gate passes.

Thus action-level feedback supplies dense supervision, trajectory-level outcome
supplies the scientific objective, and recursive harness improvement changes
the scoring/evidence mechanism by generation rather than emitting a per-action
route.

## 8. Claim boundary

The current strongest defensible claim is:

> A route-free numerical uncertainty harness can combine internal model
> instability and external scientific evidence, calibrate action worthiness,
> acquire evidence according to marginal portfolio value, and improve strict
> scientific gain under a held-out selective-risk certificate.

It is not yet defensible to claim uniform gains on every task. Current global
budget results rationally concentrate on the higher-yield pairwise task, while
task-balanced unique/extreme selection remains materially weaker and is
reported as a stress-test failure rather than hidden.
