# H14 External Source-Decomposition Pilot

Fold 0 development only. Costs assume evidence is acquired only for actions
eventually selected, so this is an upper bound for the learned VoE policy.

## Feedback-Frozen Policies

Feedback selects one method and threshold; acceptance certifies it once;
test copies it unchanged.

| family | task | method | feedback coverage | acceptance n | acceptance risk | risk UCB | deployed | test coverage | test risk | test gain |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| `direct_fused_mean` | `external_extreme_phonons` | `direct_fused_mean` | 0.0500 | 20 | 0.0000 | 0.1851 | `True` | 0.1081 | 0.0000 | +0.005057 |
| `direct_fused_mean` | `external_pairwise_dielectric` | `direct_fused_mean` | 0.0200 | 1 | 0.0000 | 0.9833 | `False` | 0.0041 | 0.0000 | +0.000247 |
| `direct_fused_mean` | `external_unique_jdft2d` | `direct_fused_mean` | 0.0500 | 6 | 0.0000 | 0.4946 | `False` | 0.0367 | 0.0000 | +0.001868 |
| `direct_high_fidelity` | `external_extreme_phonons` | `direct_high_fidelity` | 0.0500 | 20 | 0.0000 | 0.1851 | `True` | 0.1081 | 0.0000 | +0.005490 |
| `direct_high_fidelity` | `external_pairwise_dielectric` | `direct_high_fidelity` | 0.1000 | 52 | 0.4038 | 0.5605 | `False` | 0.0934 | 0.3778 | -0.003815 |
| `direct_high_fidelity` | `external_unique_jdft2d` | `direct_high_fidelity` | 0.0500 | 6 | 0.0000 | 0.4946 | `False` | 0.0367 | 0.0000 | +0.002015 |
| `matched_no_source_tail` | `external_extreme_phonons` | `no_source_tail_1.00` | 0.0500 | 20 | 0.0000 | 0.1851 | `True` | 0.1081 | 0.0000 | +0.005057 |
| `matched_no_source_tail` | `external_pairwise_dielectric` | `no_source_tail_1.00` | 0.0200 | 1 | 0.0000 | 0.9833 | `False` | 0.0041 | 0.0000 | +0.000247 |
| `matched_no_source_tail` | `external_unique_jdft2d` | `no_source_tail_1.00` | 0.0500 | 6 | 0.0000 | 0.4946 | `False` | 0.0367 | 0.0000 | +0.001868 |
| `proposed_source_tail` | `external_extreme_phonons` | `source_tail_1.00` | 0.0500 | 20 | 0.0000 | 0.1851 | `True` | 0.1081 | 0.0000 | +0.005057 |
| `proposed_source_tail` | `external_pairwise_dielectric` | `source_tail_0.50` | 0.0200 | 0 | 0.0000 | 0.0000 | `False` | 0.0041 | 0.0000 | +0.000247 |
| `proposed_source_tail` | `external_unique_jdft2d` | `source_tail_1.00` | 0.0500 | 6 | 0.0000 | 0.4946 | `False` | 0.0367 | 0.0000 | +0.001868 |

## Same-Split Grid Diagnostic

This legacy diagnostic searches acceptance prefixes and is deliberately
more conservative than the feedback-frozen protocol above.

| method | aggregate coverage | aggregate gain | pair gain | unique gain | extreme gain |
|---|---:|---:|---:|---:|---:|
| `direct_composition` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `direct_fused_mean` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `direct_high_fidelity` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `direct_structure` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_mean_0.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_mean_0.25` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_mean_0.50` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_mean_1.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_p_0.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_p_0.25` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_p_0.50` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_p_1.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_tail_0.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_tail_0.25` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_tail_0.50` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `no_source_tail_1.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_mean_0.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_mean_0.25` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_mean_0.50` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_mean_1.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_p_0.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_p_0.25` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_p_0.50` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_p_1.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_tail_0.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_tail_0.25` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_tail_0.50` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `source_tail_1.00` | 0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
