# Sci-VoI-RHI Regime-Held-Out Results

> Primary metric: oracle-normalized net scientific utility at a fixed 10% action budget; higher is better.

- Data: `20987` sanitized historical proxy records.
- Outer folds: `45` complete scientific regimes.
- Selection never reads held-out-regime test records.
- Direct LLM-as-judge is intentionally excluded.

## Aggregate Results

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Utility efficiency | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scivoi_policy_always_accept` | 0.6735 ± 0.1634 | 0.6473 | 0.1045 | 0.6389 | 0.6735 | 45 |
| `scivoi_rhi` | 0.6601 ± 0.1548 | 0.6049 | 0.2209 | 0.6256 | 0.6601 | 45 |

## Paired Comparisons

| Variant | vs | Utility Δ | Risk-adj Δ | Risk Δ | 95% CI | Win rate | Sign-test p |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |

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
| `discover_unique` | `scivoi_policy_always_accept` | 0.7662 | 0.2302 |
| `discover_unique` | `scivoi_rhi` | 0.7340 | 0.6928 |
| `extreme_properties` | `scivoi_policy_always_accept` | 0.6765 | 0.1000 |
| `extreme_properties` | `scivoi_rhi` | 0.6391 | 0.3000 |
| `matbench_pairwise` | `scivoi_policy_always_accept` | 0.6492 | 0.0747 |
| `matbench_pairwise` | `scivoi_rhi` | 0.6492 | 0.0747 |

## Interpretation

- `scivoi_policy_always_accept` is the direct RHI-style recursive update: it accepts every schema-valid executable VoI mutation, matching the original paper's no-rollback update pattern more closely than the conservative guarded variant.
- Direct Sci-VoI-RHI is the strongest non-LLM method by risk-adjusted utility and has much lower selective risk than static utility and verbal confidence baselines.
- The conservative guarded `scivoi_rhi` still improves over original reliability-only RHI and static full reliability, but it underuses beneficial utility mutations and is not the best final variant.
- Existing RHI v4/v5 results are diagnostic and are not reused for tuning this protocol.
- Historical records are offline proxies rather than online MatBot trajectories.
