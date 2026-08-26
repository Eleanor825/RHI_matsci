# h22_extreme_gap_verify_before_execute

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 64 | 32 | 0.0000 | 0.0894 | `True` | 53 | 21 | 0.0000 | +0.000616 |
| `source_voi_probability` | 0.0500 | 60 | 32 | 0.0000 | 0.0894 | `True` | 50 | 18 | 0.0000 | +0.000416 |
| `no_source_voi_expected` | 0.0500 | 65 | 32 | 0.0000 | 0.0894 | `True` | 54 | 16 | 0.0000 | +0.000111 |
| `no_source_voi_probability` | 0.0500 | 56 | 30 | 0.0000 | 0.0950 | `True` | 47 | 15 | 0.0000 | +0.000192 |
| `external_variance` | 0.0200 | 28 | 16 | 0.0000 | 0.1707 | `False` | 25 | 14 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.015947`; 
positive fraction: `0.0646`.
