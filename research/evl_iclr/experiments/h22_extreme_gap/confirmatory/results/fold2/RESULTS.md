# h22_extreme_gap_verify_before_execute

Confirmatory result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 72 | 41 | 0.0000 | 0.0705 | `True` | 77 | 42 | 0.0000 | +0.001234 |
| `source_voi_probability` | 0.1000 | 69 | 41 | 0.0000 | 0.0705 | `True` | 72 | 40 | 0.0000 | +0.001239 |
| `no_source_voi_expected` | 0.1000 | 86 | 39 | 0.0000 | 0.0739 | `True` | 97 | 38 | 0.0000 | +0.000559 |
| `no_source_voi_probability` | 0.1000 | 71 | 37 | 0.0000 | 0.0778 | `True` | 79 | 35 | 0.0000 | +0.000778 |
| `external_variance` | 0.0500 | 10 | 8 | 0.0000 | 0.3123 | `False` | 8 | 5 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.014420`; 
positive fraction: `0.0938`.
