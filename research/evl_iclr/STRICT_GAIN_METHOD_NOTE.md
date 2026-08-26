# Strict Budgeted Action-Worthiness Harness

## 1. Decision problem

A scientific agent proposes candidate actions indexed by `i`. Before execution,
the uncertainty harness observes model-internal replicas and external evidence
replicas, but not the terminal scientific outcome. The downstream laboratory
has a budget of at most `B` executions.

For action `i`, define strict realized net scientific gain

\[
G_i = S_i U_i - \lambda_c C_i
      - \lambda_h (1-S_i) H_i,
\]

where `S_i` is terminal success, `U_i` is normalized scientific utility,
`C_i` is action cost, and `H_i` is irreversible failure harm. Abstention has
gain zero.

The resource allocation problem is

\[
\max_{z_i\in\{0,1\}} \sum_i z_i\,\mathbb E[G_i\mid X_i]
\quad\text{s.t.}\quad \sum_i z_i\le B.
\]

Therefore the unconstrained optimum ranks actions by expected gain, executes
only actions with positive expected gain, and never fills unused budget with a
negative-gain action. A train-set quantile or Lagrange multiplier can be used
to solve the allocation problem, but it is not a realized scientific outcome.

This separation corrects the rejected H2 formulation, which subtracted a
top-decile reservation value from every outcome and then forced a per-task
quota. That construction conflated outcome value with resource allocation and
could force harmful actions when a task had no safe candidate.

## 2. Factorized action-worthiness distribution

For internal model replica `m` and external evidence replica `e`, the harness
estimates

\[
p_{me}=P(S_i=1\mid X_i,m,e),\qquad
\mu_{me}=\mathbb E[U_i\mid S_i=1,X_i,m,e].
\]

The crossed gain cell is

\[
\hat g_{me}=p_{me}\mu_{me}-\lambda_c C_i
-\lambda_h(1-p_{me})H_i.
\]

The model reports the crossed mean and an exact weighted two-way ANOVA
decomposition

\[
V_{\mathrm{epi}}
=V_{\mathrm{internal}}+V_{\mathrm{external}}+V_{\mathrm{interaction}}.
\]

`Internal` measures variation between model replicas after averaging external
evidence. `External` measures variation between evidence replicas after
averaging models. `Interaction` captures model-specific sensitivity to a
particular evidence view. The implementation verifies the equality
numerically rather than treating the three terms as heuristic features.

Aleatoric gain variance is estimated separately from conditional utility
residual variance and Bernoulli success uncertainty. Total predictive variance
is the sum of aleatoric and crossed epistemic components.

## 3. Calibration and conservative score

Action-worthiness is the event `G_i > 0`. Model probabilities are temperature
calibrated on a feedback subset that is disjoint from training, acceptance,
and test records.

A second feedback subset fits a one-sided normalized split-conformal lower
prediction bound

\[
L_i=\hat g_i-q_{1-\alpha}\hat\sigma_i.
\]

Under exchangeability, `L_i` has marginal lower coverage. The paper must not
claim that marginal conformal coverage alone guarantees selective risk after
ranking and selection. H3 uses `L_i` as a conservative numerical ranking score,
not as the final risk certificate.

## 4. Frozen budgeted policy

The primary policy is fixed before acceptance outcomes are inspected:

1. rank all actions jointly by `L_i`;
2. discard actions with `L_i <= 0`;
3. execute at most `ceil(0.10 N)` remaining actions.

The budget is global because task utilities are normalized on train-only
distributions and the scientific objective is total gain under a shared
resource constraint. Per-task coverage is reported as a diagnostic. A task may
receive zero executions when no action has defensible positive gain; this is a
scientific abstention, not a missing prediction.

## 5. Selective-risk guarantee

Let the frozen policy select `n` acceptance actions, of which `k` have
non-positive realized gain. The exact one-sided Clopper--Pearson upper bound is

\[
\overline R(k,n,\delta).
\]

The harness is accepted only if

\[
n>0,\quad \overline R(k,n,0.05)\le 0.10,
\quad \sum_{i\in A}G_i>0.
\]

Because the policy, score definition, budget, threshold, and risk level are
fixed before the acceptance outcomes are observed, this certificate does not
pay a multiple-testing penalty. If a family of policies were searched on the
acceptance set, a simultaneous correction would be required.

The word `exact` requires an explicit sampling condition: the selected losses
must be i.i.d. Bernoulli draws from the deployment population induced by a
fixed action-wise eligibility rule. A global batch top-k rule couples selected
actions and may produce heterogeneous error probabilities, so H3's
Clopper--Pearson result is reported as an independent held-out batch
certificate, not as an assumption-free action-wise deployment theorem. The H4
protocol removes this ambiguity by freezing an action-wise score threshold,
using Learn-then-Test correction when selecting it, and applying
outcome-independent uniform thinning only when a hard batch budget binds.

The test set is inaccessible to the acceptance-only runner. Test predictions
and outcomes are loaded only after the primary acceptance certificate passes.

## 6. Safe recursive harness improvement

RHI operates over a restricted, auditable contract language rather than free
text alone. A contract specifies:

- realized-gain target;
- allocation scope;
- numerical uncertainty score;
- evidence stages;
- budget and positivity threshold.

Trajectory failure feedback can propose mutations. For example, negative
selected gain triggers separation of realized gain from resource shadow price;
forced negative task actions trigger a global budget cap; failed risk control
triggers conformal lower-gain ranking; predictive external variance triggers a
verification evidence stage.

The proposed contract is fitted and selected only with train/feedback data. A
single frozen candidate is then compared with its predecessor on acceptance.
It replaces the predecessor only when it has positive gain and an exact risk
certificate, and improves total gain over any already-certified predecessor.

## 7. Claim hierarchy

The paper should make claims in this order:

1. the harness predicts strict action gain more accurately than static and
   direct-LLM alternatives;
2. crossed internal/external variance identifies different error sources;
3. the frozen global policy increases realized scientific gain under a fixed
   action budget;
4. independent acceptance certifies selective risk;
5. safe RHI improves the numerical harness contract without touching the base
   scientific agent;
6. controlled real-outcome replay transfers the safety behavior to historical
   LFP measurements;
7. live MatBot deployment remains necessary for a full online claim.

The paper must distinguish controlled historical replay from live MatBot
runtime and must report zero-coverage or low-coverage tasks rather than hiding
them in aggregate utility.
