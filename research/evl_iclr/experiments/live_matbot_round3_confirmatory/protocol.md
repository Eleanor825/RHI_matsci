# Live MatBot Tool Round-3 Confirmatory Protocol

Frozen on August 23, 2026, before exporting or inspecting any round-3 outcome.

## Objective

Evaluate a numerical action-worthiness harness on actually executed local
scientific tools while keeping terminal scientific outcomes historical and
held out. This experiment is complementary to the semi-synthetic H5 evidence
allocation experiment: it tests whether task-native tool outputs can be
captured, calibrated, and used without route labels.

## Strict gain

For executed action `i`,

`G_i = S_i U_i - 0.15(C_i + E_i) - 0.25(1-S_i)H_i`.

For an unexecuted action, only already acquired evidence cost is charged. The
global executor may select at most 10% of the batch.

## Tools

- `matbench_pairwise`: pymatgen composition/structure features followed by a
  train-only ExtraTrees elasticity surrogate;
- `discover_unique`: the same train-only property surrogate plus nearest-
  neighbor composition/structure novelty;
- `extreme_properties`: retrieval of the complete published candidate artifact
  followed by live RDKit MW, logP, TPSA, QED, HBA, and HBD computation against
  the published target and tolerance bounds. DRD2 remains explicitly missing.

Every tool trace records execution provenance and has
`hidden_outcome_read=false`. Terminal labels and utilities are attached only in
post-outcome events. The claim is live scientific-tool execution over
historical candidates, not new wet-lab execution.

## Numerical harness

Task-specific ExtraTrees heads estimate probability of positive gain and
expected strict execution gain from all available internal signals, external
signals, cost signals, and real tool outputs. Probability is task-specific
Platt calibrated. Expected gain receives a one-sided normalized split-conformal
lower bound at `alpha=0.10`. The harness emits numeric outputs only and never a
route.

Development uses all 3,601 round-1 feedback live-tool trajectories. Within each
task, 75% are used for fitting and 25% for calibration, selected by SHA-256 of
the frozen seed and record ID. Round-3 test is used only once for evaluation.

## Fresh test

Export 2,000 records per task from the expanded round-2 raw pool after excluding
all scientific fingerprints in both prior 24,000-record H5 export and the full
24,900-record round-2 export. Only records assigned to the pre-existing `test`
split are executed and evaluated. No label, utility, gain, or model output is
used in record selection.

## Baselines and gate

Compare against static reliability, base numerical uncertainty without live
tools, and the fixed hand-composed real-tool numerical score. The primary
learned method passes if it:

1. selects at least one action;
2. has one-sided 95% Clopper--Pearson selective-risk UCB at most 0.10;
3. has positive strict trajectory gain after charging evidence for every test
   action;
4. exceeds every listed baseline in strict trajectory gain;
5. has no higher empirical selective risk than the strongest listed baseline.

Task breakdown and task-balanced stress tests are reported without hiding zero
coverage or weak tasks.

## Frozen implementation hashes

- `real_scientific_tools.py`:
  `5138afd84cdbee7c195befc3c6b61f49e1b7c9396452a4cece7f2cc3f2667561`
- `capture_live_matbot_tool_trajectories.py`:
  `5597779cf8a35cdc0c40a07ed60695d13f5cf936762f472c9a74631f29c1ba9a`
- `evaluate_learned_live_matbot_harness.py`:
  `d1b4d65df2d365ce95701d7df1ff0b6e91b638e2707b3bc10d34fd93cb538349`
- `matbot_strict_gain_runtime.py`:
  `a60949fd6c28ead9fe393a17d0b64d71ff59aafd11294ead6e7f33b4afaf9d73`
