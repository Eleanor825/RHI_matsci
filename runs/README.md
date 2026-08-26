# Experiment Runs Index

This directory intentionally keeps both positive and negative experiment
artifacts so method revisions can be compared without overwriting prior
results. All reported numbers are offline historical/proxy results, not online
MatBot trajectory outcomes.

## Current Material-Data Artifacts

| Run | Purpose | Key result |
| --- | --- | --- |
| `label_audit_matbench_pairwise_v1/` | Label/utility/leakage audit for the new real-material Matbench A/B preference task. | `8,000` records, `28` regimes, positive rate `0.5534`; labels and utilities are exactly recomputable from hidden Matbench property values; visible leakage count is `0`. |
| `label_audit_main_materials_v1/` | Combined audit for the updated main material tasks: `matbench_pairwise`, `discover_unique`, and `extreme_properties`. | `20,987` records across `45` regimes; label consistency, utility consistency, and visible leakage checks all pass. |
| `matbench_pairwise_smoke_v1/` | One-seed smoke check that the VoI suite runs on the new real-material pairwise preference task. | `scivoi_rhi` reaches net utility `0.8048 ± 0.2225` over `28` held-out crystal-system-pair folds; this is not yet the full paper-scale rerun. |

## Preserved Positive Legacy Runs

| Run | Purpose | Key result |
| --- | --- | --- |
| `scivoi_rhi_v1/` | Legacy Sci-VoI-RHI held-out-regime evaluation over the earlier PBO + unique-material + extreme-property setup. | Best direct RHI-style VoI variant reaches net utility `0.6443 ± 0.2052`, risk-adjusted utility `0.6043`, and selective risk `0.1600` over 105 held-out-regime folds. Needs rerun on the updated main materials benchmark. |
| `mechanism_ablation_v2/` | Clean mechanism ablation for utility, uncertainty, routing, cost, and recursive updates without component bundle loops on the earlier setup. | `scivoi_policy_always_accept` beats frozen `scivoi_policy_never_accept` by `+0.0454` utility and `-0.3522` selective risk; utility-only beats reliability-only by `+0.0447`. |

## Preserved Comparison Runs

| Run | Purpose | Why it is kept |
| --- | --- | --- |
| `rhi_experiments_v4/` | Five-seed reliability-only RHI evaluation on the earlier 15,717-record PBO-including setup. | Diagnostic negative result: reliability-only RHI is comparable to learned baselines but does not establish a positive self-improvement claim. |
| `rhi_experiments_v5_evolution/` | Self-evolution checkpoint ablation for `H0`–`H3` under guarded and always-accept policies. | Shows recursive mutation is active, but aggregate H0-to-H3 primary score worsens; useful contrast for why Sci-VoI is needed. |
| `related_work_baselines_v1/` | Non-LLM related-work baseline sweep over confidence, evidence, self-consistency proxies, semantic-entropy proxy, ensembles, and acquisition-style policies. | Shows many raw-utility baselines are high-risk; merged comparison keeps Sci-VoI-RHI best by risk-adjusted utility (`0.6043`) with much lower risk (`0.1600`). |
| `paper_bootstrap_v1_multiseed/` | First paper-derived weak-label bootstrap on 500 electrolyte action records across five seeds. | Demonstrates feasibility of historical-paper action labels and records domain/group-shift limitations. |
| `label_audit_v1/` | Full historical label/utility consistency and leakage audit. | Confirms the proxy labels and utilities are internally reproducible and that oracle fields are withheld from visible text. |
| `direct_judge_baseline_v1/` | One-shot direct LLM-as-judge baseline protocol. | Preserves the baseline definition without publishing mock scores; generated score caches are intentionally not committed. |
| `direct_judge_smoke_v1/` | End-to-end smoke run for the configured `gpt-5.5` judge client. | Confirms the real provider path works after adding gateway-safe request headers; not treated as a formal benchmark result. |
| `direct_judge_subset500_v1/` | Balanced 500-record historical subset scored by `gpt-5.5` direct judge. | Provides a real LLM-as-judge subset comparison against verbal-confidence and evidence-heuristic baselines; not a replacement for a full main-materials run. |
| `hybrid_llm_judge_subset_v1/` | Cached `gpt-5.5` direct judge fused with the local Sci-VoI harness on the same 500-record subset. | Shows hybrid calibration improves diagnostic score (`0.3315` vs `0.3602`) and ECE (`0.0974` vs `0.1548`) over direct `gpt-5.5`, but not Risk@10% or hit rate; still calls the LLM for every action. |
| `direct_judge_subset100_gpt56luna_v1/` | Balanced 100-record subset prepared for `gpt-5.6-luna` direct judge. | Documents that `gpt-5.6-luna` is currently blocked by provider `429`; includes a same-100 cached `gpt-5.5` control with score `0.3484` and Risk@10% `0.2000`. |

## Reading Order

For the current experiment inventory and data protocol, start with:

0. `docs/research/CURRENT_EXPERIMENTS_AND_SELF_EVOLUTION.md` for the current
   experiment inventory and signal-aware training loop.
1. `benchmarks/materials_action_trajectories_v1/README.md` for the complete
   action-level benchmark and grouped trajectory schema.

For paper writing or result comparison, then read:

2. `label_audit_main_materials_v1/README.md` for the updated main-materials data audit.
3. `label_audit_matbench_pairwise_v1/README.md` for the new real-material A/B preference construction audit.
4. `matbench_pairwise_smoke_v1/README.md` for the first executable check on the new pairwise task.
5. `scivoi_rhi_main_materials_v1/README.md` for the current complete materials result.
6. `related_work_baselines_v1/SCIVOI_COMPARISON.md` for the combined related-work baseline table on the earlier setup.
7. `rhi_experiments_v5_evolution/RESULTS.md` for the prior self-evolution ablation.
8. `paper_bootstrap_v1_multiseed/README.md` for historical-paper weak-label evidence.
9. `direct_judge_subset500_v1/README.md` for the current real `gpt-5.5` subset result.
10. `hybrid_llm_judge_subset_v1/README.md` for the cached LLM + VoI subset comparison.

## Claim Boundaries

- Positive legacy claim supported here: Sci-VoI-style harness optimization improves
  offline proxy action-value decisions over reliability-only RHI and static full
  reliability baselines on the earlier PBO-including setup.
- Updated materials-data claim supported here: the repository now constructs and
  audits a real Matbench A/B materials preference task and a 20,987-record
  three-task main materials benchmark.
- The complete current-materials Sci-VoI result is now preserved in
  `scivoi_rhi_main_materials_v1/README.md`; it supports the offline
  risk-adjusted result, not an online MatBot claim.
- Claim not yet supported here: online MatBot scientific discovery improves in
  real lab/DFT trajectories.
- Important limitation: labels and utilities are reconstructed from published or
  benchmark-derived outcomes, not expert annotations of live agent decisions.
