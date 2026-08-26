# H15 Powered Unique Two-Stage VoE

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0200 | 33 | 84 | 0.1905 | 0.2748 | `False` | 43 | 98 | 0.2143 | +0.000000 |
| `source_voi_probability` | 0.0500 | 85 | 87 | 0.1034 | 0.1736 | `True` | 87 | 99 | 0.1717 | +0.000985 |
| `no_source_voi_expected` | 0.0200 | 52 | 84 | 0.1667 | 0.2482 | `False` | 47 | 92 | 0.1957 | +0.000000 |
| `no_source_voi_probability` | 0.0500 | 98 | 83 | 0.1084 | 0.1816 | `True` | 100 | 97 | 0.1443 | +0.000967 |
| `external_variance` | 0.0500 | 121 | 86 | 0.0814 | 0.1474 | `True` | 129 | 98 | 0.1122 | +0.000832 |

Feedback realized VoE mean: `-0.015961`; 
positive fraction: `0.0683`.
