# h23_superconductivity_verify_before_execute

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 309 | 155 | 0.0452 | 0.0831 | `True` | 329 | 165 | 0.0000 | +0.000586 |
| `source_voi_probability` | 0.1000 | 309 | 155 | 0.0645 | 0.1070 | `False` | 329 | 165 | 0.0000 | +0.000000 |
| `no_source_voi_expected` | 0.0500 | 155 | 62 | 0.0000 | 0.0472 | `True` | 165 | 66 | 0.0000 | +0.000243 |
| `no_source_voi_probability` | 0.1000 | 309 | 155 | 0.2129 | 0.2742 | `False` | 329 | 165 | 0.0000 | +0.000000 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016791`; 
positive fraction: `0.0514`.
