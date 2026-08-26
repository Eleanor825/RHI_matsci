# h17_verify_before_execute_development

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 177 | 87 | 0.0000 | 0.0338 | `True` | 164 | 89 | 0.0000 | +0.001198 |
| `source_voi_probability` | 0.1000 | 193 | 91 | 0.0000 | 0.0324 | `True` | 177 | 86 | 0.0000 | +0.000971 |
| `no_source_voi_expected` | 0.1000 | 239 | 106 | 0.0000 | 0.0279 | `True` | 220 | 96 | 0.0000 | +0.000831 |
| `no_source_voi_probability` | 0.1000 | 236 | 105 | 0.0000 | 0.0281 | `True` | 216 | 96 | 0.0000 | +0.000867 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.013364`; 
positive fraction: `0.0986`.
