# h19_phonons_frozen_external_validation

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 76 | 58 | 0.1034 | 0.1940 | `True` | 147 | 113 | 0.0354 | +0.002492 |
| `source_voi_probability` | 0.1000 | 71 | 57 | 0.1053 | 0.1972 | `True` | 140 | 111 | 0.0360 | +0.002555 |
| `no_source_voi_expected` | 0.1000 | 69 | 57 | 0.1053 | 0.1972 | `True` | 142 | 111 | 0.0360 | +0.002514 |
| `no_source_voi_probability` | 0.1000 | 68 | 57 | 0.1053 | 0.1972 | `True` | 141 | 110 | 0.0364 | +0.002504 |
| `source_voi_tail` | 0.0500 | 8 | 8 | 0.0000 | 0.3123 | `False` | 50 | 41 | 0.0488 | +0.000000 |
| `no_source_voi_tail` | 0.1000 | 66 | 57 | 0.1053 | 0.1972 | `True` | 141 | 111 | 0.0360 | +0.002530 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016656`; 
positive fraction: `0.0615`.
