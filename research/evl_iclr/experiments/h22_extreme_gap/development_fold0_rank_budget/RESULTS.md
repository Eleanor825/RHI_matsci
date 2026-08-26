# h22_extreme_gap_verify_before_execute

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 87 | 44 | 0.0000 | 0.0658 | `True` | 93 | 47 | 0.0426 | +0.000418 |
| `source_voi_probability` | 0.1000 | 87 | 44 | 0.0000 | 0.0658 | `True` | 93 | 47 | 0.0638 | +0.000201 |
| `no_source_voi_expected` | 0.1000 | 87 | 44 | 0.0000 | 0.0658 | `True` | 93 | 47 | 0.1489 | -0.001863 |
| `no_source_voi_probability` | 0.1000 | 87 | 44 | 0.0000 | 0.0658 | `True` | 93 | 47 | 0.1277 | -0.001323 |
| `external_variance` | 0.0500 | 44 | 18 | 0.0000 | 0.1533 | `False` | 47 | 19 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.015947`; 
positive fraction: `0.0646`.
