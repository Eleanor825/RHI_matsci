# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001650 |
| `source_voi_probability` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001650 |
| `no_source_voi_expected` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001662 |
| `no_source_voi_probability` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001650 |
| `source_voi_tail` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001629 |
| `no_source_voi_tail` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001655 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016421`; 
positive fraction: `0.0580`.
