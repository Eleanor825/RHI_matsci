# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 104 | 42 | 0.0000 | 0.0688 | `True` | 110 | 44 | 0.0000 | +0.000495 |
| `source_voi_probability` | 0.0500 | 104 | 42 | 0.0000 | 0.0688 | `True` | 110 | 44 | 0.0000 | +0.000480 |
| `no_source_voi_expected` | 0.0500 | 104 | 42 | 0.0000 | 0.0688 | `True` | 110 | 44 | 0.0000 | +0.000484 |
| `no_source_voi_probability` | 0.0500 | 104 | 42 | 0.0000 | 0.0688 | `True` | 110 | 44 | 0.0000 | +0.000484 |
| `source_voi_tail` | 0.0500 | 104 | 42 | 0.0000 | 0.0688 | `True` | 110 | 44 | 0.0000 | +0.000457 |
| `no_source_voi_tail` | 0.0500 | 104 | 42 | 0.0000 | 0.0688 | `True` | 110 | 44 | 0.0000 | +0.000495 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016697`; 
positive fraction: `0.0606`.
