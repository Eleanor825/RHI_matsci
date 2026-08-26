# Related-Work Positioning

## Confidence and uncertainty for language models

Semantic entropy and black-box uncertainty methods estimate dispersion among
semantically distinct generations and use it to predict answer correctness
(`Kuhn et al., 2023, arXiv:2302.09664`; `Lin et al., 2023,
arXiv:2305.19187`). The computational-confidence view studies whether answer
logit differences act as readouts of a latent decision variable (`Kumaran et
al., 2026, arXiv:2607.12447`). These methods provide useful internal signals,
but their target is confidence that an answer is correct.

EVL-Harness instead estimates the strict marginal scientific gain of executing
an action. Correctness confidence is only one input. Utility, evidence cost,
action cost, reversibility, failure harm, source risk, OOD, and the possibility
that additional evidence changes the selected portfolio are part of the target
and decision rule.

## Uncertainty in multi-step agents

SAUP propagates step-level uncertainty through an LLM-agent trajectory with
situation-dependent weighting (`Zhao et al., 2024, arXiv:2412.01033`). Related
belief-control work detects deviation of an agent's internal belief from the
environment state and truncates uninformative RL trajectories. These methods
address how uncertainty or belief error accumulates across steps.

EVL-Harness addresses an orthogonal question at each decision point: whether a
specific externally consequential action has positive calibrated scientific
gain, and whether evidence is worth buying before execution. It uses both
action-level and trajectory-level feedback, but does not reduce trajectory
uncertainty to a propagated scalar.

## Value of information and uncertainty-aware planning

Classical and modern uncertainty-aware planning uses value of information,
exploration bonuses, ensembles, or model uncertainty to decide where an agent
should explore or whether a learned model should be trusted. The closest
conceptual connection is the value of acquiring information before acting.

Our distinction is the portfolio object. Evidence is valued by the expected
change in the strict top-k scientific portfolio relative to the exact
leave-one-out replacement action, after charging incremental evidence cost.
Raw uncertainty is not automatically valuable: evidence receives value only
when it is expected to change a budget-constrained scientific decision.

## Selective prediction and statistical risk control

Risk-controlling prediction sets, Learn-then-Test, and Conformal Risk Control
provide model-agnostic finite-sample tools for calibrating prediction systems
(`Bates et al., 2021, arXiv:2101.02703`; `Angelopoulos et al., 2021,
arXiv:2110.01052`; `Angelopoulos et al., 2022, arXiv:2208.02814`). EVL-Harness
uses split calibration and held-out testing in this tradition.

The new decision object is action execution under a shared budget. Because
top-k ranking couples actions, we deliberately avoid claiming an action-wise
distribution-free theorem. The confirmatory statement is a one-sided
Clopper--Pearson certificate for the selected set on an independent held-out
batch, together with a separately reported split-conformal gain lower bound.

## Harness self-improvement

Recursive Harness Self-Improvement represents the agent harness as a prompt-
level loop and iteratively revises it using pairwise feedback over revision
history (`Lee et al., 2026, arXiv:2607.15524`). Self-Harness similarly mines
weaknesses from traces, proposes harness edits, and accepts them after
regression testing (`Zhang et al., 2026, arXiv:2606.09498`). Their primary
objective is downstream task performance and execution-trace quality.

Our recursive object is narrower and statistically grounded: the numerical
contract mapping internal/external evidence to action gain, uncertainty,
calibration, and evidence value. A revision is accepted only when a disjoint
acceptance set satisfies a frozen selective-risk and strict-net-gain gate. This
turns harness evolution from prompt-quality optimization into risk-controlled
scientific decision-layer optimization.

## Positioning sentence

> Existing work estimates whether an answer is correct, propagates uncertainty
> through a trajectory, or improves the agent loop. EVL-Harness estimates
> whether a scientific action is worth executing, values evidence by its
> marginal effect on a budgeted scientific portfolio, and recursively improves
> that numerical decision contract under held-out risk control.

## Strongest novelty boundary

The paper should not claim that confidence estimation, conformal calibration,
ANOVA decomposition, value of information, or harness evolution is individually
new. The contribution is their decision-theoretic composition around a new
target—strict action-level scientific gain—and the corresponding benchmark,
live-tool trajectory contract, and confirmatory evaluation regime.
