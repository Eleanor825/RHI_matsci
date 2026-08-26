# h18_dielectric_verify_before_execute_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 216 | 140 | 0.0643 | 0.1095 | `True` | 311 | 239 | 0.0628 | +0.002525 |
| `source_voi_probability` | 0.1000 | 280 | 164 | 0.0610 | 0.1012 | `True` | 335 | 250 | 0.0720 | +0.002471 |
| `no_source_voi_expected` | 0.1000 | 218 | 141 | 0.0638 | 0.1087 | `True` | 319 | 242 | 0.0702 | +0.002466 |
| `no_source_voi_probability` | 0.1000 | 220 | 142 | 0.0634 | 0.1080 | `True` | 320 | 242 | 0.0661 | +0.002482 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.015283`; 
positive fraction: `0.0768`.
