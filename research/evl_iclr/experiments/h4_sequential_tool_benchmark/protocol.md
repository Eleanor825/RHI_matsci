# H4 Protocol: Leakage-Audited Sequential Tool Benchmark v2

Date locked: 2026-08-21

## Motivation

The v1 historical benchmark removes hidden outcomes correctly, but it also
removes task-defining pre-execution measurements.  This makes many actions
unidentifiable from the visible state and yields zero risk-controlled
coverage even for nonlinear predictors.  H4 tests whether an explicit,
costed tool-observation process restores identifiability without exposing the
evaluation outcome.

## Decision Object

For each proposed scientific action, the harness returns only

```text
q_t = P(G_execute > 0 | base state, observations through step t)
```

It does not select a route.  A separate evaluation policy may acquire a tool
observation or execute the proposed action.  Tool costs are subtracted when
trajectory-level net gain is reported.

## Tasks

1. `matbench_pairwise`: choose the candidate predicted to have higher bulk
   modulus; exact Materials Project values remain hidden.
2. `discover_unique`: advance a material that jointly has high performance
   and train-library novelty; exact discovery score and top-decile label
   remain hidden.
3. `extreme_properties`: advance a generated molecule toward a target under
   seven property constraints; exact generated property row and target-hit
   label remain hidden.

## Observation Process

Each action has three evidence stages:

1. `base`: the original leakage-sanitized prompt state.
2. `screening`: a deterministic-seed, lower-fidelity task-specific tool
   observation with declared uncertainty and cost.
3. `verification`: an independent higher-fidelity observation with smaller
   declared uncertainty and larger cost.

For this controlled benchmark, tool outputs are noisy observations centered
on the historical hidden measurement.  The observation mechanism is fixed
before evaluation and is explicitly marked `semi_synthetic`.  It is not
claimed to be a live MatBot trajectory.  Independent observation seeds form
the external-replica axis.

## Leakage Rules

- Hidden labels, exact utilities, exact discovery scores, exact pairwise
  gaps, exact generated property rows, and exact target-hit indicators cannot
  appear in visible text, visible features, or tool provenance.
- Tool observations may report noisy task-native measurements, their declared
  standard deviations, and derived pre-execution probabilities.
- Task thresholds and normalizers are fit on the training split only.
- Test scientific regimes are group-held-out.
- A structural leakage audit must fail if any forbidden field or exact hidden
  value is copied into the visible observation.

## Model and Source Decomposition

Three independently seeded ExtraTrees gain regressors form the internal
replica axis.  Three independent tool-observation seeds form the external
axis.  Every action must have a complete 3 x 3 prediction matrix.

For gain predictions `g[m,e]`:

```text
U_internal = mean_e Var_m(g[m,e])
U_external = Var_e mean_m(g[m,e])
```

A task-conditional source-decomposed calibrator is fit only on the feedback
split.  Risk thresholds are selected only on the acceptance split.

## Confirmatory Comparisons

- base-only static reliability model;
- screening evidence without source variance;
- verification evidence without source variance;
- SD-Worth with mean gain plus internal/external variance;
- oracle ranking, reported only as an upper bound.

## Primary Endpoints

At `alpha in {0.10, 0.20, 0.30}`:

1. exact-binomial risk-controlled coverage on the acceptance-selected test
   threshold;
2. test selective risk;
3. selected mean budget-conditioned scientific gain;
4. trajectory net gain after subtracting acquired tool cost;
5. worst-task and worst-held-out-regime selective risk.

Calibration endpoints are Brier score, ECE, and AURC.

## Confirmatory Success Criteria

H4 is supported only if all conditions hold on at least four of five seeds:

1. verification-stage SD-Worth has nonzero exact risk-controlled test coverage
   at `alpha <= 0.20`;
2. selected mean gain and trajectory net gain are both positive;
3. SD-Worth improves risk-controlled net gain over base-only static
   reliability and the variance-free verification model;
4. verification improves Brier score or AURC over base evidence;
5. the leakage audit passes for every generated observation.

If only the controlled observation channel succeeds, the result establishes
benchmark identifiability and mechanism validity, not live-MatBot validity.

## Seeds and Splits

- Seeds: `1, 7, 13, 21, 42`.
- Split policy: existing deterministic four-way split with held-out complete
  scientific groups for test and disjoint train/feedback/acceptance records.
- Budget fraction: `0.10`.
- Scientific gain configuration: repository default, with zero-gain outside
  option.

