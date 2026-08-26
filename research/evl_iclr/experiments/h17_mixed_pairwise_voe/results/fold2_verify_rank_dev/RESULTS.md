# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001842 |
| `source_voi_probability` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001820 |
| `no_source_voi_expected` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001898 |
| `no_source_voi_probability` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001926 |
| `source_voi_tail` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001884 |
| `no_source_voi_tail` | 0.1000 | 207 | 104 | 0.0000 | 0.0284 | `True` | 220 | 110 | 0.0000 | +0.001889 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016689`; 
positive fraction: `0.0587`.
