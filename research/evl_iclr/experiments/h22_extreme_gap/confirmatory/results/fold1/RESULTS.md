# h22_extreme_gap_verify_before_execute

Confirmatory result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 79 | 44 | 0.0000 | 0.0658 | `True` | 91 | 63 | 0.0000 | +0.001663 |
| `source_voi_probability` | 0.1000 | 78 | 45 | 0.0000 | 0.0644 | `True` | 82 | 56 | 0.0000 | +0.001552 |
| `no_source_voi_expected` | 0.0500 | 39 | 28 | 0.0000 | 0.1015 | `False` | 41 | 31 | 0.0000 | +0.000000 |
| `no_source_voi_probability` | 0.0500 | 39 | 28 | 0.0000 | 0.1015 | `False` | 42 | 31 | 0.0000 | +0.000000 |
| `external_variance` | 0.0100 | 16 | 9 | 0.0000 | 0.2831 | `False` | 16 | 9 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.015364`; 
positive fraction: `0.0692`.
