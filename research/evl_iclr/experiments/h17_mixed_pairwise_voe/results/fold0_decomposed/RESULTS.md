# H15 Powered Unique Two-Stage VoE

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0100 | 19 | 89 | 0.2360 | 0.3217 | `False` | 28 | 104 | 0.2596 | +0.000000 |
| `source_voi_probability` | 0.0100 | 20 | 94 | 0.2872 | 0.3737 | `False` | 31 | 103 | 0.2718 | +0.000000 |
| `no_source_voi_expected` | 0.0200 | 53 | 82 | 0.1707 | 0.2540 | `False` | 55 | 93 | 0.2043 | +0.000000 |
| `no_source_voi_probability` | 0.0500 | 105 | 88 | 0.1023 | 0.1717 | `True` | 106 | 98 | 0.1531 | +0.000839 |
| `external_variance` | 0.0500 | 121 | 86 | 0.0814 | 0.1474 | `True` | 129 | 98 | 0.1122 | +0.000832 |

Feedback realized VoE mean: `-0.015961`; 
positive fraction: `0.0683`.
