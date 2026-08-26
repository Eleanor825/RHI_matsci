# h18_dielectric_verify_before_execute_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 117 | 92 | 0.1087 | 0.1774 | `True` | 180 | 123 | 0.1463 | +0.000704 |
| `source_voi_probability` | 0.0500 | 116 | 93 | 0.1183 | 0.1882 | `True` | 170 | 116 | 0.1552 | +0.000710 |
| `no_source_voi_expected` | 0.0500 | 146 | 116 | 0.1293 | 0.1921 | `True` | 219 | 137 | 0.1533 | +0.000559 |
| `no_source_voi_probability` | 0.0500 | 108 | 87 | 0.0920 | 0.1598 | `True` | 167 | 114 | 0.1491 | +0.000673 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016681`; 
positive fraction: `0.0480`.
