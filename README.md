# harness_matsci

Runtime uncertainty harness for scientific agents in materials discovery.

## What it does

- Turns each scientific action into an `ActionRecord` with visible context, evidence, feature signals, and a binary label for whether the action was worth executing.
- Trains a transparent `LogisticGate` to estimate `P(action worth doing)`.
- Routes low-confidence actions to `retrieve_more`, `simulate`, `ask_expert`, or `abstain` instead of forcing a bad proceed.
- Ships three main offline materials benchmarks aligned with our paper targets: Matbench material pairwise preference, DiSCoVeR-style unique-material discovery, and RL-CC extreme-property discovery.
- Provides an RHI-MatSci loop that evolves the scientific agent's prompt-level contracts and orchestrator hops from trajectory feedback, then accepts or rejects each revision on held-out actions.
- Adds Sci-VoI-RHI, an executable value-of-information harness that estimates reliability, scientific utility, epistemic uncertainty, verification value, and action cost before routing the next step.
- Supports a strict internal/external uncertainty contract, including answer-logit difference features for models that expose next-token logits; see `runs/signal_contract_audit_v1/` for current data coverage.
- Packages the main benchmark as complete action-level decision units and grouped trajectories with an explicit decision-time input/outcome boundary.

## Data format

Use JSONL with one `ActionRecord` per line. Key fields:

- `visible_context`: what the agent saw
- `candidate_action`: the proposed step
- `features`: uncertainty and cost signals such as `evidence_support`, `ood_score`, `tool_agreement`, `cost`, `reversibility`
- `label`: `1` if the step was worth doing, else `0`
- `utility`: optional downstream gain for discovery metrics

## Data provenance

The main offline materials benchmark contains 20,987 paper-artifact /
benchmark-derived action proxies from three public materials / chemical task
families: Matbench material pairwise preference, DiSCoVeR-style unique-material
screening with Matbench `log10(K_VRH)`, and RL-CC extreme-property molecule
generation. It is not live MatBot trajectory logging and not manual extraction
of PDF actions. The older González et al. PBO data is retained only as an
auxiliary controlled optimization sanity check, not as a materials benchmark.

Full data lineage, public URLs, label equations, leakage controls, examples,
and reproduction commands are documented in
`docs/research/DATA_PROVENANCE_AND_ACTION_CONSTRUCTION.md`.

## Commands

Generate a benchmark dataset:

```bash
python -m harness_matsci make-benchmark --benchmark preferential_bo --out data/preferential_bo.jsonl
```

Generate the real-material Matbench pairwise preference data:

```bash
PYTHONPATH=src python3 -m harness_matsci.matbench_pairwise \
  --source-path /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks/cache/matbench_log_kvrh.json.bz2 \
  --out /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks/matbench_pairwise_actions.jsonl \
  --n-pairs 8000 \
  --seed 1729 \
  --min-true-gap 0.003 \
  --update-summary
```

Train a gate:

```bash
python -m harness_matsci train --data data/preferential_bo.jsonl --out artifacts/gate.json
```

Evaluate a saved gate:

```bash
python -m harness_matsci evaluate --data data/preferential_bo.jsonl --model artifacts/gate.json --out reports/eval.json
```

Run a benchmark campaign:

```bash
python -m harness_matsci campaign --out reports/campaign.json
```

Run trajectory-feedback Recursive Harness Self-Improvement:

```bash
python -m harness_matsci rhi \
  --data data/preferential_bo.jsonl \
  --iterations 3 \
  --out runs/rhi_preferential_bo/report.json
```

The default proposer is deterministic and reproducible. An OpenAI-compatible
proposer can be supplied from Python through `JSONLLMHarnessProposer`; its
candidate is still schema-validated and must pass the held-out acceptance gate.

Run the three paper experiments:

```bash
python -m harness_matsci experiment-suite \
  --out runs/rhi_experiments_v4/summary.json \
  --markdown-out runs/rhi_experiments_v4/README.md \
  --data-dir /path/to/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --n-per-task 300
```

The suite reports: (1) single-task main results, (2) leave-one-task-out transfer,
(3) balanced joint multi-task training with per-task slices, and (4) self-evolution
checkpoint ablations. Each experiment is paired with a short
design note explaining what it proves and the main reviewer challenge.

Run the self-evolution ablation without an external model:

```bash
python -m harness_matsci experiment-suite \
  --experiments 4 \
  --data-dir /path/to/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --rhi-iterations 3 \
  --out runs/rhi_experiments_v5_evolution/summary.json \
  --markdown-out runs/rhi_experiments_v5_evolution/README.md
```

Experiment 4 reports untouched-test performance for active `H0`–`H3` checkpoints
under both guarded acceptance and an always-accept mutation control.

Run the non-LLM Sci-VoI-RHI held-out-regime suite:

```bash
python -m harness_matsci voi-experiment-suite \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --epochs 60 \
  --iterations 3 \
  --out runs/scivoi_rhi_v1/summary.json \
  --markdown-out runs/scivoi_rhi_v1/README.md
```

This suite excludes the direct LLM-as-judge baseline. It evaluates 21 complete
scientific regimes held out one at a time over five seeds, with component and
acceptance-policy ablations.

Package the complete action-level materials benchmark:

```bash
PYTHONPATH=src python3 -m harness_matsci build-trajectory-benchmark \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --tasks matbench_pairwise,discover_unique,extreme_properties \
  --out-dir benchmarks/materials_action_trajectories_v1
```

This produces 20,987 action decisions and 45 grouped benchmark-regime
trajectories. Train or run the gate from `model_input`; reserve `outcome` and
`raw_record` for feedback, labels, and audits. The exact schema is documented
in `benchmarks/materials_action_trajectories_v1/README.md`.

Generate the first prompt-conditioned trajectory run (300 instances per task):

```bash
PYTHONPATH=src python3 -m harness_matsci generate-prompt-trajectories \
  --data-dir /Users/huan/matsci_uncertainty/data/raw/material_discovery_tasks \
  --out-dir runs/prompt_trajectory_main_v1 \
  --n-per-task 300 \
  --max-steps 3
```

The current reproducible run uses a deterministic prompt policy; it stores the
full prompt and action menu and is deliberately marked `llm_generated=false`.
This isolates trajectory schema and leakage before replacing the policy with a
real LLM backend.

Run the combined action-level plus trajectory-outcome RHI experiment:

```bash
PYTHONPATH=src python3 -m harness_matsci evaluate-combined-trajectory-rhi \
  --trajectories runs/prompt_trajectory_main_v1/trajectories.jsonl \
  --out-dir runs/prompt_trajectory_main_v1/combined_rhi \
  --iterations 3 \
  --epochs 45
```

Each action contributes local calibration/evidence feedback. The complete
trajectory additionally contributes wrong-commit, missed-opportunity,
failed-recovery, cost, and route-length outcomes. These signals are aggregated
once per RHI round to propose one harness mutation; actions do not mutate the
harness independently. The fixed-budget checkpoint table is written to
`runs/prompt_trajectory_main_v1/combined_rhi/README.md`.

Evaluate fixed baselines on the same trajectory set:

```bash
PYTHONPATH=src python3 -m harness_matsci evaluate-prompt-baselines \
  --trajectories runs/prompt_trajectory_main_v1/trajectories.jsonl \
  --out-dir runs/prompt_trajectory_main_v1/baselines \
  --budget-fraction 0.10
```

Run the non-LLM related-work baseline sweep without rerunning recursive RHI
mutations:

```bash
python -m harness_matsci voi-experiment-suite \
  --data-dir /path/to/material_discovery_tasks \
  --methods random_policy,cost_only,verbal_confidence,evidence_heuristic,tool_agreement,self_consistency_proxy,semantic_entropy_proxy,cost_aware_confidence,h0_reliability,static_full_reliability,ensemble_reliability,ensemble_lcb,static_utility,utility_ucb,utility_lcb,uncertainty_sampling,static_voi \
  --components '' \
  --acceptance-policies '' \
  --seeds 1,7,13,21,42 \
  --epochs 60 \
  --out runs/related_work_baselines_v1/summary.json \
  --markdown-out runs/related_work_baselines_v1/README.md
```

This sweep covers offline analogues of confidence judges, self-consistency,
semantic entropy, ensemble uncertainty, selective prediction, and BO-style
acquisition policies. True direct LLM judge and multi-call agentic judge
baselines still require API calls.

Enable the optional one-shot LLM direct-as-judge baseline:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.5
export OPENAI_BASE_URL=https://www.hi-code.cc
export OPENAI_REASONING_EFFORT=xhigh
python -m harness_matsci experiment-suite \
  --data-dir /path/to/material_discovery_tasks \
  --direct-judge-model "$OPENAI_MODEL" \
  --direct-judge-base-url "$OPENAI_BASE_URL" \
  --direct-judge-reasoning-effort "$OPENAI_REASONING_EFFORT" \
  --direct-judge-cache runs/direct_judge_cache/scores.json \
  --out runs/rhi_experiments_v5_direct_judge/summary.json \
  --markdown-out runs/rhi_experiments_v5_direct_judge/README.md
```

`llm_direct_judge` is a one-shot baseline: the LLM receives only the visible
context, candidate action, and pre-execution evidence, returns `p_success`, and
does not train, receive trajectory feedback, or mutate a harness. Its threshold
is calibrated on the validation/feedback partition, while the test partition is
used only for final evaluation. Calls are cached by model and prompt version;
the repository does not include API keys or generated score caches.

Run the cached hybrid LLM + VoI subset experiment:

```bash
PYTHONPATH=src python3 -m harness_matsci.hybrid_judge \
  --records runs/direct_judge_subset500_v1/records.jsonl \
  --judge-cache runs/direct_judge_cache/subset500_gpt55_scores.json \
  --local-train-fraction 0.5 \
  --out-dir runs/hybrid_llm_judge_subset_v1
```

This hybrid experiment uses cached `gpt-5.5` one-shot judge scores as a
semantic sensor and evaluates whether local VoI calibration/blending improves
the action-worthiness score. It is a 500-record subset result, not the full
main-materials held-out-regime protocol.

Run the first historical-paper bootstrap experiment:

```bash
python -m harness_matsci paper-bootstrap \
  --data /Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl \
  --workdir runs/paper_bootstrap_v1
```

## RHI-MatSci loop

The implementation follows the reusable RHI mechanism from
`alphaXiv/recursive-harness-self-improvement` without copying its repository
task code:

```text
H_i -> action trajectories -> failure feedback -> proposer mutation
    -> schema validation -> held-out H_i/H_{i+1} comparison -> accept/rollback
```

The evolved state includes `required_features`, scientific agent roles and
contracts, routing gates, and orchestrator hops. The base material-science
agent is not fine-tuned. The default proposer combines internal signals
(confidence, perturbation stability, and disagreement) with external signals
(evidence support/conflict, source reliability, tool agreement, and OOD), then
promotes signals whose feedback-split error association is high. Candidate
revisions are trained on train/feedback data and selected on held-out
acceptance shards; the test split is untouched until final evaluation.

## Current results

The experiment index in `runs/README.md` lists the current Sci-VoI result and
preserves earlier comparison runs so reliability-only RHI, self-evolution
ablations, paper-bootstrap data, and direct-judge protocol artifacts remain
easy to compare.

It now also includes the full label/utility audit in `runs/label_audit_v1/`
and the cleaner mechanism ablation in `runs/mechanism_ablation_v2/`.

The reliability-only five-seed snapshots in `runs/rhi_experiments_v4/RESULTS.md`
and `runs/rhi_experiments_v5_evolution/RESULTS.md` remain important diagnostic
references. They show that the original reliability-only RHI mutation is active
but does not produce a positive H0-to-H3 improvement.

The preserved positive method result is Sci-VoI-RHI in
`runs/scivoi_rhi_v1/README.md`. That run used the earlier 15,717-record setup
that included auxiliary PBO. The direct RHI-style executable VoI variant
(`scivoi_policy_always_accept`) achieved net utility `0.6443 ± 0.2052` and
selective risk `0.1600` over 105 paired held-out-regime folds. The main
materials benchmark has since been updated to replace PBO with
`matbench_pairwise`. The current-materials result is preserved in
`runs/scivoi_rhi_main_materials_v1/README.md`: 20,987 records, 45 regimes, and
225 held-out-regime/seed folds. `scivoi_policy_always_accept` reaches net
utility `0.7655 ± 0.2041`, risk-adjusted utility `0.7233`, and selective risk
`0.1688`; guarded `scivoi_rhi` reaches `0.7633`, `0.7096`, and `0.2148`.

The related-work baseline sweep in
`runs/related_work_baselines_v1/SCIVOI_COMPARISON.md` adds random/cost sanity
checks, confidence/evidence judges, agreement and semantic-entropy proxies,
ensemble lower-confidence bounds, and acquisition-style utility policies. The
strongest raw-utility heuristics (`verbal_confidence`, `tool_agreement`,
`uncertainty_sampling`) are high-risk, while Sci-VoI-RHI remains best by
risk-adjusted utility (`0.6043`) with much lower selective risk (`0.1600`).

In the tables, `non_rhi_seed` means a single learned logistic gate using the
initial RHI feature contract but with no recursive harness mutation or
trajectory-conditioned acceptance. `static_full` is a single learned gate with
the full available feature set. `verbal_confidence` is not an LLM judge; it is a
text/feature heuristic. The direct LLM comparison is implemented separately and
must be run with an explicitly configured API model.

The current real direct-judge result is a 500-record `gpt-5.5` subset in
`runs/direct_judge_subset500_v1/README.md`: score `0.3602`, Risk@10% `0.3462`,
hit rate `0.6800`, and ECE `0.1548`. This strong one-shot judge beats the
current quick same-subset Sci-VoI score on top-k ranking. This result should be
reported separately from both the preserved legacy full run and the updated
20,987-record main materials setup. A 100-record
`gpt-5.6-luna` attempt is documented in
`runs/direct_judge_subset100_gpt56luna_v1/README.md`; the provider currently
returns HTTP `429`, so it is not counted as a completed model result. Gateway
smoke checks currently pass for `gpt-5.5`, `gpt-5.4`, and `gpt-5.4-mini`.

The cached hybrid LLM + VoI subset result is in
`runs/hybrid_llm_judge_subset_v1/README.md`. On the same 500-record subset,
`hybrid_static_blend` improves the diagnostic score from `0.3602` to `0.3315`
and ECE from `0.1548` to `0.0974` versus direct `gpt-5.5`, but it does not yet
improve Risk@10% or hit rate and still calls the LLM for every action.

The current related-work baseline checklist is in
`docs/research/RELATED_WORK_BASELINES.md`.

For presentation-ready summaries, use
`docs/research/EXPERIMENT_RESULTS_TABLES.md` for compact result tables and
`docs/research/LEADERSHIP_BRIEF.md` for the current leadership update.

## Design note

This repository reuses the harness idea from recursive self-improvement systems,
but the task layer is scientific: the model learns whether a materials action is
worth executing, verifying, or stopping, not whether code changes are good.

## Latest Direct-Judge Comparison

The latest paired control reuses 500 records with real cached GPT-5.5
one-shot-judge scores and evaluates both methods under the same top-10% action
budget. On the RHI-held-out split (74 test actions, 8 selected actions),
GPT-5.5 direct judge obtains Risk@10% `0.6250`, hit rate `0.3750`, and mean
utility `0.008380`; task-conditioned RHI obtains Risk@10% `0.7500`, hit rate
`0.2500`, and mean utility `0.007865`. The mutation is not accepted on this
small split. This is a negative/neutral control, not a claim that RHI beats the
LLM judge. Details are in `runs/direct_judge_baseline_v1/README.md` and
`runs/direct_judge_subset500_v2/paired_summary.json`.

## Graph-RHI experiment

A graph-structured variant is implemented in `src/harness_matsci/graph_rhi.py`.
It keeps a fixed seed harness `H0`, evaluates two independent mutations (`H1`
evidence/source and `H2` OOD/stability), and permits a true `H1+H2 -> H3`
merge only when both branches independently pass the held-out acceptance rule.
The five-fold result is in `runs/graph_rhi_external_five_fold_v1/README.md`.

Reproduce the external Graph-RHI branching/merge pilot:

```bash
PYTHONPATH=src python3 -m harness_matsci.cli graph-rhi-external \
  --pairwise-root benchmarks/external_h17_pairwise \
  --unique-root benchmarks/external_h15_unique \
  --extreme-root benchmarks/external_h14 \
  --folds 5 --epochs 80 \
  --out-dir runs/graph_rhi_external_five_fold_v1
```

Graph-RHI nodes are complete harness versions, not isolated feature branches.
H0/H1/H2 share the agent interface, action unit, calibration, acceptance rule,
and budget. Each branch modifies a different harness component; H3 performs a
conflict-checked component-wise merge only when both parent harnesses pass
acceptance. The current pilot is in `runs/graph_rhi_external_five_fold_v2/`.

## Full-data Graph-RHI result

The complete `materials_action_trajectories_v1` benchmark was evaluated with
Graph-RHI: `20,987` action records across `45` grouped trajectories. The
trajectory-disjoint split contains `13,148` train, `2,648` feedback, `2,622`
acceptance, and `2,569` test actions. Under the fixed top-10% action budget,
H0 obtains risk `0.2996`, hit rate `0.7004`, and mean utility `0.1387`; H1
evidence/source obtains risk `0.2957`, hit rate `0.7043`, and mean utility
`0.1471`; H2 OOD/stability obtains risk `0.2879`, hit rate `0.7121`, and mean
utility `0.1541`. H2 passes acceptance and becomes active, H1 is rejected, and
no true two-parent H3 merge is activated. Full output is in
`runs/graph_rhi_full_v2/`.
