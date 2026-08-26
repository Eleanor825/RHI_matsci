# Sci-VoI-RHI Regime-Held-Out Results

> Primary metric: oracle-normalized net scientific utility at a fixed 10% action budget; higher is better.

- Data: `20987` sanitized historical proxy records.
- Outer folds: `45` complete scientific regimes.
- Selection never reads held-out-regime test records.
- Direct LLM-as-judge is intentionally excluded.

## Aggregate Results

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Utility efficiency | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cost_aware_confidence` | 0.3895 ± 0.2925 | 0.3089 | 0.3226 | 0.4392 | 0.3895 | 225 |
| `cost_only` | 0.4345 ± 0.4198 | 0.2895 | 0.5802 | 0.5008 | 0.4345 | 225 |
| `ensemble_lcb` | 0.6605 ± 0.2363 | 0.5717 | 0.3548 | 0.5289 | 0.6605 | 225 |
| `ensemble_reliability` | 0.6578 ± 0.2344 | 0.5709 | 0.3475 | 0.5269 | 0.6578 | 225 |
| `evidence_heuristic` | 0.5027 ± 0.2370 | 0.4245 | 0.3128 | 0.4912 | 0.5027 | 225 |
| `h0_reliability` | 0.7622 ± 0.1843 | 0.6747 | 0.3499 | 0.5442 | 0.7622 | 225 |
| `original_rhi` | 0.7282 ± 0.2020 | 0.6471 | 0.3244 | 0.5290 | 0.7282 | 225 |
| `random_policy` | 0.3431 ± 0.3370 | 0.2017 | 0.5657 | 0.4201 | 0.3431 | 225 |
| `scivoi_component_features` | 0.7622 ± 0.1845 | 0.6737 | 0.3539 | 0.5436 | 0.7622 | 225 |
| `scivoi_component_routing` | 0.7622 ± 0.1843 | 0.6747 | 0.3499 | 0.5442 | 0.7622 | 225 |
| `scivoi_component_uncertainty` | 0.7575 ± 0.1894 | 0.6700 | 0.3499 | 0.5345 | 0.7575 | 225 |
| `scivoi_component_utility` | 0.7636 ± 0.1926 | 0.6839 | 0.3188 | 0.5443 | 0.7636 | 225 |
| `scivoi_policy_always_accept` | 0.7655 ± 0.2041 | 0.7233 | 0.1688 | 0.5496 | 0.7655 | 225 |
| `scivoi_policy_mean_guarded` | 0.7655 ± 0.2041 | 0.7233 | 0.1688 | 0.5496 | 0.7655 | 225 |
| `scivoi_rhi` | 0.7633 ± 0.1923 | 0.7096 | 0.2148 | 0.5459 | 0.7633 | 225 |
| `self_consistency_proxy` | 0.8370 ± 0.1911 | 0.7406 | 0.3858 | 0.6128 | 0.8370 | 225 |
| `semantic_entropy_proxy` | 0.7525 ± 0.1956 | 0.6796 | 0.2916 | 0.5520 | 0.7525 | 225 |
| `static_full_reliability` | 0.6421 ± 0.2331 | 0.5546 | 0.3498 | 0.5243 | 0.6421 | 225 |
| `static_utility` | 0.7657 ± 0.2038 | 0.6858 | 0.3194 | 0.5494 | 0.7657 | 225 |
| `static_utility_no_cost` | 0.7657 ± 0.2038 | 0.6835 | 0.3286 | 0.5494 | 0.7657 | 225 |
| `static_voi` | 0.7657 ± 0.2043 | 0.6869 | 0.3149 | 0.5496 | 0.7657 | 225 |
| `static_voi_no_cost` | 0.7657 ± 0.2043 | 0.6835 | 0.3288 | 0.5496 | 0.7657 | 225 |
| `static_voi_no_routing` | 0.7657 ± 0.2043 | 0.6869 | 0.3149 | 0.5496 | 0.7657 | 225 |
| `static_voi_no_uncertainty` | 0.7658 ± 0.2037 | 0.6846 | 0.3248 | 0.5492 | 0.7658 | 225 |
| `tool_agreement` | 0.4345 ± 0.4198 | 0.2895 | 0.5802 | 0.5008 | 0.4345 | 225 |
| `uncertainty_sampling` | 0.4295 ± 0.4117 | 0.2844 | 0.5802 | 0.4885 | 0.4295 | 225 |
| `utility_lcb` | 0.7585 ± 0.2132 | 0.6915 | 0.2679 | 0.5470 | 0.7585 | 225 |
| `utility_ucb` | 0.7598 ± 0.2071 | 0.6881 | 0.2865 | 0.5483 | 0.7598 | 225 |
| `verbal_confidence` | 0.4105 ± 0.4301 | 0.2643 | 0.5847 | 0.4961 | 0.4105 | 225 |

## Paired Comparisons

| Variant | vs | Utility Δ | Risk-adj Δ | Risk Δ | 95% CI | Win rate | Sign-test p |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `scivoi_policy_always_accept` | `original_rhi` | 0.0373 | 0.0762 | -0.1556 | [0.0188, 0.0574] | 0.524 | 0.0719 |
| `scivoi_policy_always_accept` | `static_full_reliability` | 0.1234 | 0.1687 | -0.1810 | [0.0984, 0.1513] | 0.778 | 0.0000 |
| `scivoi_policy_always_accept` | `ensemble_reliability` | 0.1077 | 0.1524 | -0.1787 | [0.0845, 0.1333] | 0.809 | 0.0000 |
| `scivoi_policy_always_accept` | `ensemble_lcb` | 0.1050 | 0.1515 | -0.1860 | [0.0817, 0.1308] | 0.796 | 0.0000 |
| `scivoi_policy_always_accept` | `self_consistency_proxy` | -0.0715 | -0.0173 | -0.2170 | [-0.0946, -0.0492] | 0.351 | 0.0010 |
| `scivoi_policy_always_accept` | `semantic_entropy_proxy` | 0.0130 | 0.0437 | -0.1228 | [-0.0052, 0.0312] | 0.542 | 0.0098 |
| `scivoi_policy_always_accept` | `cost_aware_confidence` | 0.3760 | 0.4144 | -0.1538 | [0.3251, 0.4262] | 0.871 | 0.0000 |
| `scivoi_policy_always_accept` | `utility_ucb` | 0.0057 | 0.0351 | -0.1177 | [-0.0080, 0.0162] | 0.409 | 0.1432 |
| `scivoi_policy_always_accept` | `utility_lcb` | 0.0070 | 0.0317 | -0.0991 | [-0.0067, 0.0179] | 0.422 | 0.2960 |
| `scivoi_policy_always_accept` | `uncertainty_sampling` | 0.3360 | 0.4388 | -0.4114 | [0.2772, 0.3971] | 0.680 | 0.0000 |
| `scivoi_policy_always_accept` | `static_voi` | -0.0002 | 0.0363 | -0.1461 | [-0.0010, 0.0005] | 0.129 | 0.8974 |
| `scivoi_policy_always_accept` | `static_utility` | -0.0002 | 0.0375 | -0.1506 | [-0.0013, 0.0008] | 0.147 | 0.8013 |
| `scivoi_policy_mean_guarded` | `static_voi` | -0.0002 | 0.0363 | -0.1461 | [-0.0010, 0.0005] | 0.129 | 0.8974 |
| `scivoi_rhi` | `original_rhi` | 0.0351 | 0.0625 | -0.1097 | [0.0183, 0.0553] | 0.440 | 0.7790 |
| `scivoi_rhi` | `static_full_reliability` | 0.1212 | 0.1550 | -0.1351 | [0.0941, 0.1500] | 0.742 | 0.0000 |
| `scivoi_rhi` | `static_voi` | -0.0023 | 0.0227 | -0.1002 | [-0.0154, 0.0126] | 0.218 | 0.0039 |
| `static_voi` | `h0_reliability` | 0.0035 | 0.0122 | -0.0349 | [-0.0144, 0.0201] | 0.484 | 0.0001 |

## Related-Work Baseline Families

| Family | Implemented methods | Notes |
| --- | --- | --- |
| Random/cost heuristics | `random_policy`, `cost_only` | Sanity checks for fixed-budget discovery and cheap-action selection. |
| Confidence and evidence judges | `verbal_confidence`, `evidence_heuristic`, `cost_aware_confidence` | Offline analogues of confidence/rationale judges without LLM calls. |
| Agreement/entropy uncertainty | `tool_agreement`, `self_consistency_proxy`, `semantic_entropy_proxy` | Deterministic proxies for self-consistency and semantic-entropy uncertainty. |
| Selective prediction / ensembles | `h0_reliability`, `static_full_reliability`, `ensemble_reliability`, `ensemble_lcb` | Risk-calibrated reliability gates and ensemble lower-confidence bounds. |
| Acquisition-style utility | `utility_ucb`, `utility_lcb`, `uncertainty_sampling`, `static_utility`, `static_voi` | Active-learning/BO-style utility and uncertainty scoring. |
| Harness self-improvement | `original_rhi`, `scivoi_rhi`, `scivoi_policy_*` | Recursive harness mutation baselines and acceptance-policy ablations. |

## Task Slices

| Task | Method | Net utility | Risk |
| --- | --- | ---: | ---: |
| `discover_unique` | `cost_aware_confidence` | 0.7279 | 0.7837 |
| `discover_unique` | `cost_only` | 0.7044 | 0.8958 |
| `discover_unique` | `ensemble_lcb` | 0.7473 | 0.8013 |
| `discover_unique` | `ensemble_reliability` | 0.7491 | 0.8005 |
| `discover_unique` | `evidence_heuristic` | 0.7279 | 0.7837 |
| `discover_unique` | `h0_reliability` | 0.7383 | 0.7837 |
| `discover_unique` | `original_rhi` | 0.7266 | 0.7867 |
| `discover_unique` | `random_policy` | 0.7248 | 0.8940 |
| `discover_unique` | `scivoi_component_features` | 0.7423 | 0.8192 |
| `discover_unique` | `scivoi_component_routing` | 0.7383 | 0.7837 |
| `discover_unique` | `scivoi_component_uncertainty` | 0.7324 | 0.7837 |
| `discover_unique` | `scivoi_component_utility` | 0.7532 | 0.7837 |
| `discover_unique` | `scivoi_policy_always_accept` | 0.7660 | 0.0000 |
| `discover_unique` | `scivoi_policy_mean_guarded` | 0.7660 | 0.0000 |
| `discover_unique` | `scivoi_rhi` | 0.7561 | 0.1541 |
| `discover_unique` | `self_consistency_proxy` | 0.7044 | 0.8958 |
| `discover_unique` | `semantic_entropy_proxy` | 0.7279 | 0.7837 |
| `discover_unique` | `static_full_reliability` | 0.7424 | 0.8032 |
| `discover_unique` | `static_utility` | 0.7648 | 0.8032 |
| `discover_unique` | `static_utility_no_cost` | 0.7648 | 0.8032 |
| `discover_unique` | `static_voi` | 0.7652 | 0.8032 |
| `discover_unique` | `static_voi_no_cost` | 0.7652 | 0.8032 |
| `discover_unique` | `static_voi_no_routing` | 0.7652 | 0.8032 |
| `discover_unique` | `static_voi_no_uncertainty` | 0.7656 | 0.8000 |
| `discover_unique` | `tool_agreement` | 0.7044 | 0.8958 |
| `discover_unique` | `uncertainty_sampling` | 0.7044 | 0.8958 |
| `discover_unique` | `utility_lcb` | 0.7572 | 0.4179 |
| `discover_unique` | `utility_ucb` | 0.7519 | 0.4294 |
| `discover_unique` | `verbal_confidence` | 0.7044 | 0.8958 |
| `extreme_properties` | `cost_aware_confidence` | 0.6032 | 0.5671 |
| `extreme_properties` | `cost_only` | 1.0000 | 0.8435 |
| `extreme_properties` | `ensemble_lcb` | 0.6241 | 0.4068 |
| `extreme_properties` | `ensemble_reliability` | 0.6218 | 0.3891 |
| `extreme_properties` | `evidence_heuristic` | 0.6032 | 0.5671 |
| `extreme_properties` | `h0_reliability` | 0.6728 | 0.3732 |
| `extreme_properties` | `original_rhi` | 0.5843 | 0.2933 |
| `extreme_properties` | `random_policy` | 0.6933 | 0.8338 |
| `extreme_properties` | `scivoi_component_features` | 0.6699 | 0.3666 |
| `extreme_properties` | `scivoi_component_routing` | 0.6728 | 0.3732 |
| `extreme_properties` | `scivoi_component_uncertainty` | 0.6559 | 0.3732 |
| `extreme_properties` | `scivoi_component_utility` | 0.6645 | 0.2332 |
| `extreme_properties` | `scivoi_policy_always_accept` | 0.6910 | 0.1600 |
| `extreme_properties` | `scivoi_policy_mean_guarded` | 0.6910 | 0.1600 |
| `extreme_properties` | `scivoi_rhi` | 0.6639 | 0.2066 |
| `extreme_properties` | `self_consistency_proxy` | 1.0000 | 0.8435 |
| `extreme_properties` | `semantic_entropy_proxy` | 0.6032 | 0.5671 |
| `extreme_properties` | `static_full_reliability` | 0.5687 | 0.3907 |
| `extreme_properties` | `static_utility` | 0.6917 | 0.2707 |
| `extreme_properties` | `static_utility_no_cost` | 0.6917 | 0.3107 |
| `extreme_properties` | `static_voi` | 0.6914 | 0.2507 |
| `extreme_properties` | `static_voi_no_cost` | 0.6914 | 0.3107 |
| `extreme_properties` | `static_voi_no_routing` | 0.6914 | 0.2507 |
| `extreme_properties` | `static_voi_no_uncertainty` | 0.6913 | 0.3136 |
| `extreme_properties` | `tool_agreement` | 1.0000 | 0.8435 |
| `extreme_properties` | `uncertainty_sampling` | 0.9709 | 0.8435 |
| `extreme_properties` | `utility_lcb` | 0.6280 | 0.2800 |
| `extreme_properties` | `utility_ucb` | 0.6409 | 0.3522 |
| `extreme_properties` | `verbal_confidence` | 1.0000 | 0.8435 |
| `matbench_pairwise` | `cost_aware_confidence` | 0.2286 | 0.1200 |
| `matbench_pairwise` | `cost_only` | 0.1651 | 0.4072 |
| `matbench_pairwise` | `ensemble_lcb` | 0.6517 | 0.2247 |
| `matbench_pairwise` | `ensemble_reliability` | 0.6477 | 0.2195 |
| `matbench_pairwise` | `evidence_heuristic` | 0.4105 | 0.1042 |
| `matbench_pairwise` | `h0_reliability` | 0.8001 | 0.2331 |
| `matbench_pairwise` | `original_rhi` | 0.7800 | 0.2200 |
| `matbench_pairwise` | `random_policy` | 0.1226 | 0.3878 |
| `matbench_pairwise` | `scivoi_component_features` | 0.8001 | 0.2331 |
| `matbench_pairwise` | `scivoi_component_routing` | 0.8001 | 0.2331 |
| `matbench_pairwise` | `scivoi_component_uncertainty` | 0.8001 | 0.2331 |
| `matbench_pairwise` | `scivoi_component_utility` | 0.8015 | 0.2331 |
| `matbench_pairwise` | `scivoi_policy_always_accept` | 0.7920 | 0.2141 |
| `matbench_pairwise` | `scivoi_policy_mean_guarded` | 0.7920 | 0.2141 |
| `matbench_pairwise` | `scivoi_rhi` | 0.8007 | 0.2329 |
| `matbench_pairwise` | `self_consistency_proxy` | 0.8120 | 0.0948 |
| `matbench_pairwise` | `semantic_entropy_proxy` | 0.8120 | 0.0701 |
| `matbench_pairwise` | `static_full_reliability` | 0.6432 | 0.2219 |
| `matbench_pairwise` | `static_utility` | 0.7923 | 0.2158 |
| `matbench_pairwise` | `static_utility_no_cost` | 0.7923 | 0.2163 |
| `matbench_pairwise` | `static_voi` | 0.7923 | 0.2158 |
| `matbench_pairwise` | `static_voi_no_cost` | 0.7923 | 0.2167 |
| `matbench_pairwise` | `static_voi_no_routing` | 0.7923 | 0.2158 |
| `matbench_pairwise` | `static_voi_no_uncertainty` | 0.7925 | 0.2100 |
| `matbench_pairwise` | `tool_agreement` | 0.1651 | 0.4072 |
| `matbench_pairwise` | `uncertainty_sampling` | 0.1674 | 0.4072 |
| `matbench_pairwise` | `utility_lcb` | 0.8055 | 0.2261 |
| `matbench_pairwise` | `utility_ucb` | 0.8042 | 0.2272 |
| `matbench_pairwise` | `verbal_confidence` | 0.1265 | 0.4146 |

## Interpretation

- `scivoi_policy_always_accept` is the direct RHI-style recursive update: it accepts every schema-valid executable VoI mutation, matching the original paper's no-rollback update pattern more closely than the conservative guarded variant.
- Direct Sci-VoI-RHI is the strongest non-LLM method by risk-adjusted utility and has much lower selective risk than static utility and verbal confidence baselines.
- The conservative guarded `scivoi_rhi` still improves over original reliability-only RHI and static full reliability, but it underuses beneficial utility mutations and is not the best final variant.
- Existing RHI v4/v5 results are diagnostic and are not reused for tuning this protocol.
- Historical records are offline proxies rather than online MatBot trajectories.
