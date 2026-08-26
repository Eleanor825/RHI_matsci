# Prompt-Trajectory Baselines

Same fixed trajectory set: `900` records; budget: `10%` per task for ranking baselines.

| Method | Coverage | Risk ↓ | Hit rate ↑ | Net utility/action ↑ | Total net utility/trajectory ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `always_execute` | 1.000 | 0.941 | 0.059 | -0.0783 | -0.0783 |
| `random_top10` | 0.100 | 0.978 | 0.022 | -0.0935 | -0.0093 |
| `verbal_confidence_top10` | 0.100 | 0.967 | 0.033 | -0.0907 | -0.0091 |
| `evidence_heuristic_top10` | 0.100 | 0.700 | 0.300 | -0.0094 | -0.0009 |
| `static_reliability_top10` | 0.100 | 0.656 | 0.344 | -0.0074 | -0.0007 |
| `prompt_policy` | 0.367 | 0.839 | 0.161 | -0.0083 | -0.0031 |
| `oracle_top10_reference` | 0.100 | 0.422 | 0.578 | 0.1253 | 0.0125 |

`oracle_top10_reference` is an upper-bound reference and is not an executable baseline.
`prompt_policy` is the deterministic prompt-conditioned construction policy, not the proposed Sci-VoI-RHI.
Direct LLM judge and Sci-VoI-RHI trajectory replay are intentionally marked pending rather than assigned mock scores.
