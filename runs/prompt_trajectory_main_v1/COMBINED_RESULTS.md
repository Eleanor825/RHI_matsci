# Prompt-Trajectory Combined Results

Fixed evaluation set: 900 prompt-conditioned trajectories, 300 per task. Ranking baselines use a per-task top-10% budget. `runtime_route` rows use the actual route emitted by the policy.

## Runtime-route comparison

| Method | Coverage ↑ | Selective risk ↓ | Hit rate ↑ | Net utility/action ↑ | Net utility/trajectory ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `always_execute` | 1.000 | 0.941 | 0.059 | -0.0783 | -0.0783 |
| `prompt_policy` | 0.367 | 0.839 | 0.161 | -0.0083 | -0.0031 |
| `static_sci_voi` | 0.064 | 0.914 | 0.086 | 0.0155 | 0.0010 |
| `scivoi_rhi` | 0.047 | 0.857 | 0.143 | 0.0213 | 0.0010 |

## Fixed-budget comparison

| Method | Coverage ↑ | Selective risk ↓ | Hit rate ↑ | Net utility/action ↑ | Net utility/trajectory ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_top10` | 0.100 | 0.978 | 0.022 | -0.0935 | -0.0093 |
| `verbal_confidence_top10` | 0.100 | 0.967 | 0.033 | -0.0907 | -0.0091 |
| `evidence_heuristic_top10` | 0.100 | 0.700 | 0.300 | -0.0094 | -0.0009 |
| `static_reliability_top10` | 0.100 | 0.656 | 0.344 | -0.0074 | -0.0007 |
| `static_sci_voi_top10` | 0.100 | 0.956 | 0.044 | -0.0892 | -0.0089 |
| `scivoi_rhi_top10` | 0.100 | 0.944 | 0.056 | -0.0861 | -0.0086 |
| `oracle_top10_reference` | 0.100 | 0.422 | 0.578 | 0.1253 | 0.0125 |

## Interpretation boundary

- These are the first prompt-conditioned trajectory results, not the final paper result.
- The deterministic `prompt_policy` is a construction baseline, not the proposed harness.
- The Sci-VoI models were trained on the remaining historical records after removing the 900 sampled trajectory records; the sampled records were used only as held-out trajectory evaluation.
- The current result does not show Sci-VoI superiority yet. The signal is that Sci-VoI-RHI has lower runtime coverage and lower risk than the deterministic prompt policy, but its risk remains high and the fixed-budget ranking is not competitive with the static reliability baseline.
- Direct LLM judge is pending because the currently configured provider route does not expose a usable model for this run; no mock score is included.
