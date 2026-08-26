# h23_superconductivity_verify_before_execute

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.0000 | +0.001693 |
| `source_voi_probability` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.0000 | +0.001579 |
| `no_source_voi_expected` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.0000 | +0.001690 |
| `no_source_voi_probability` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.0000 | +0.001665 |
| `external_variance` | 0.0500 | 155 | 62 | 0.5806 | 0.6870 | `False` | 165 | 66 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.015907`; 
positive fraction: `0.0760`.
