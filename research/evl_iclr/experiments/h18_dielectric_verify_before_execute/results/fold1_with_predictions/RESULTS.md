# h18_dielectric_verify_before_execute_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 283 | 128 | 0.0000 | 0.0231 | `True` | 310 | 149 | 0.0000 | +0.000440 |
| `source_voi_probability` | 0.1000 | 257 | 120 | 0.0000 | 0.0247 | `True` | 295 | 146 | 0.0000 | +0.000499 |
| `no_source_voi_expected` | 0.1000 | 261 | 123 | 0.0000 | 0.0241 | `True` | 287 | 146 | 0.0000 | +0.000544 |
| `no_source_voi_probability` | 0.1000 | 260 | 122 | 0.0000 | 0.0243 | `True` | 284 | 147 | 0.0000 | +0.000578 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.014423`; 
positive fraction: `0.0966`.
