# H14 Tail-Risk Real-Tool Ranking Pilot

Development-only result on consumed fold 0. Tool acquisition is assumed only
for selected actions in the cost column; learned VoE is not yet included.

| source | task | score | best prefix | precision | selected gain | population gain after cost |
|---|---|---|---|---:|---:|---:|
| `no_source` | `matbench_pairwise` | `p_positive_gain` | `top_0.050` | 0.8875 | +0.0467 | +0.002241 |
| `no_source` | `matbench_pairwise` | `mean_gain` | `top_0.050` | 0.8875 | +0.0467 | +0.002241 |
| `no_source` | `matbench_pairwise` | `tail_risk_score` | `top_0.050` | 0.8875 | +0.0467 | +0.002241 |
| `no_source` | `discover_unique` | `p_positive_gain` | `top_0.005` | 0.8000 | -0.1677 | -0.000869 |
| `no_source` | `discover_unique` | `mean_gain` | `top_0.005` | 0.8000 | -0.1677 | -0.000869 |
| `no_source` | `discover_unique` | `tail_risk_score` | `top_0.005` | 0.8000 | -0.1677 | -0.000869 |
| `no_source` | `extreme_properties` | `p_positive_gain` | `top_0.050` | 0.8000 | +0.0255 | +0.001173 |
| `no_source` | `extreme_properties` | `mean_gain` | `top_0.050` | 0.8000 | +0.0255 | +0.001173 |
| `no_source` | `extreme_properties` | `tail_risk_score` | `top_0.050` | 0.8000 | +0.0255 | +0.001173 |
| `source` | `matbench_pairwise` | `p_positive_gain` | `top_0.050` | 0.9000 | +0.0472 | +0.002264 |
| `source` | `matbench_pairwise` | `mean_gain` | `top_0.050` | 0.9000 | +0.0480 | +0.002306 |
| `source` | `matbench_pairwise` | `tail_risk_score` | `top_0.050` | 0.9000 | +0.0472 | +0.002264 |
| `source` | `discover_unique` | `p_positive_gain` | `top_0.005` | 0.8000 | -0.1677 | -0.000869 |
| `source` | `discover_unique` | `mean_gain` | `top_0.005` | 0.8000 | -0.1677 | -0.000869 |
| `source` | `discover_unique` | `tail_risk_score` | `top_0.005` | 0.8000 | -0.1677 | -0.000869 |
| `source` | `extreme_properties` | `p_positive_gain` | `top_0.050` | 0.8000 | +0.0255 | +0.001173 |
| `source` | `extreme_properties` | `mean_gain` | `top_0.050` | 0.8000 | +0.0255 | +0.001173 |
| `source` | `extreme_properties` | `tail_risk_score` | `top_0.050` | 0.8000 | +0.0255 | +0.001173 |
