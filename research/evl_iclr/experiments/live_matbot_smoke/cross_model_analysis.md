# Live MatBot Cross-Model Analysis

## Scope

- Live trajectories: 3
- Distinct model backends: 2
- Trajectories with observed terminal outcomes: 0
- Tool calls/errors: 59/11 (error rate 0.186)

## Numeric Outputs

| Field | Per-model values | Mean | Range | Population variance |
|---|---:|---:|---:|---:|
| `p_success` | 0.8000, 0.8400, 0.9000 | 0.8467 | 0.1000 | 0.001689 |
| `expected_scientific_utility` | 0.9200, 0.9500, 0.9100 | 0.9267 | 0.0400 | 0.000289 |
| `reported_internal_uncertainty` | 0.2400, 0.4300, 0.4200 | 0.3633 | 0.1900 | 0.007622 |
| `reported_external_uncertainty` | 0.9000, 0.6600, 0.7800 | 0.7800 | 0.2400 | 0.009600 |
| `expected_value_of_evidence` | 0.9500, 0.9400, 0.9400 | 0.9433 | 0.0100 | 0.000022 |
| `irreversible_risk` | 0.1200, 0.0800, 0.0400 | 0.0800 | 0.0800 | 0.001067 |
| `expected_net_scientific_gain` | 0.2520, 0.1652, 0.4350 | 0.2841 | 0.2698 | 0.012646 |

## Action Disagreement

- Pair count: 3
- Mean/min character-bigram Jaccard: 0.2083 / 0.1711
- Mean/min token Jaccard: 0.2084 / 0.1592

## Within-Model Repeated Sampling

- `mimorouter-openai/gpt-5.6-luna` (2 samples):
  - action token Jaccard=0.1791; character-bigram Jaccard=0.1921
  - `p_success` range=0.0600, population variance=0.000900
  - `expected_scientific_utility` range=0.0400, population variance=0.000400
  - `reported_internal_uncertainty` range=0.0100, population variance=0.000025
  - `reported_external_uncertainty` range=0.1200, population variance=0.003600
  - `expected_value_of_evidence` range=0.0000, population variance=0.000000
  - `irreversible_risk` range=0.0400, population variance=0.000400
  - `expected_net_scientific_gain` range=0.2698, population variance=0.018198

## Valid Interpretation

- The dispersion above is **between different model backends**. It is not within-model internal variance.
- The models' `reported_internal_uncertainty` and `reported_external_uncertainty` fields remain uncalibrated self-reports until repeated sampling, controlled evidence interventions, and observed outcomes are available.
- These trajectories validate real runtime capture and tool traces, but not scientific benefit: all 3 trajectories lack a post-outcome event.
