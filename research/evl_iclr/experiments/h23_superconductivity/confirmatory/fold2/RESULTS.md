# h23_superconductivity_verify_before_execute

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 155 | 62 | 0.0000 | 0.0472 | `True` | 165 | 66 | 0.0000 | +0.000489 |
| `source_voi_probability` | 0.0200 | 62 | 31 | 0.0000 | 0.0921 | `True` | 66 | 33 | 0.0000 | +0.000383 |
| `no_source_voi_expected` | 0.0500 | 155 | 62 | 0.0000 | 0.0472 | `True` | 165 | 66 | 0.0000 | +0.000388 |
| `no_source_voi_probability` | 0.0200 | 62 | 31 | 0.0000 | 0.0921 | `True` | 66 | 33 | 0.0000 | +0.000258 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016551`; 
positive fraction: `0.0492`.
