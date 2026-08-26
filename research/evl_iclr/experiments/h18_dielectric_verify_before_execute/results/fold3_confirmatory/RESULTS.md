# h18_dielectric_verify_before_execute_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 271 | 149 | 0.0000 | 0.0199 | `True` | 141 | 86 | 0.0000 | +0.001098 |
| `source_voi_probability` | 0.1000 | 291 | 156 | 0.0000 | 0.0190 | `True` | 157 | 95 | 0.0000 | +0.001156 |
| `no_source_voi_expected` | 0.1000 | 287 | 157 | 0.0000 | 0.0189 | `True` | 150 | 90 | 0.0000 | +0.001124 |
| `no_source_voi_probability` | 0.1000 | 295 | 158 | 0.0000 | 0.0188 | `True` | 155 | 93 | 0.0000 | +0.001116 |
| `external_variance` | 0.0200 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.014692`; 
positive fraction: `0.0812`.
