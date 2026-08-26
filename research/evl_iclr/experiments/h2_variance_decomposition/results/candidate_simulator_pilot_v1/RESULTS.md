# H2.1 Closed-Model × Retrieval-Tool Pilot

Complete matrix: `True`; actions: `30`; cells: `270`.

## Probability quality

| Score | Brier | ECE | AURC | Accuracy |
|---|---:|---:|---:|---:|
| base_ensemble | 0.2033 | 0.2261 | 0.8643 | 0.7333 |
| all_tool_views_ensemble | 0.1599 | 0.1508 | 0.8541 | 0.8333 |

## Source components

| Scope | N | Internal var | External var | Corr internal/error | Corr external/error | Tool shifts |
|---|---:|---:|---:|---:|---:|---|
| overall | 30 | 0.00191 | 0.01546 | 0.0677 | -0.1089 | `{"tool::candidate_simulator": -0.046122544444444455, "tool::nearest_analogs": -0.15359392222222223}` |
| matbench_pairwise | 10 | 0.00325 | 0.04182 | -0.1656 | -0.2780 | `{"tool::candidate_simulator": -0.16691023333333332, "tool::nearest_analogs": -0.45185823333333336}` |
| discover_unique | 10 | 0.00240 | 0.00366 | 0.7386 | 0.8490 | `{"tool::candidate_simulator": 0.0028941333333333376, "tool::nearest_analogs": -0.029710999999999998}` |
| extreme_properties | 10 | 0.00009 | 0.00091 | -0.1215 | -0.2936 | `{"tool::candidate_simulator": 0.025648466666666626, "tool::nearest_analogs": 0.020787466666666643}` |
