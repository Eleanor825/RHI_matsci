# h15_powered_unique_two_stage_voe

Confirmatory result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 101 | 95 | 0.0105 | 0.0490 | `True` | 116 | 104 | 0.0192 | +0.001724 |
| `source_voi_probability` | 0.1000 | 214 | 151 | 0.0596 | 0.1017 | `False` | 253 | 177 | 0.0565 | +0.000000 |
| `no_source_voi_expected` | 0.1000 | 194 | 147 | 0.0748 | 0.1208 | `False` | 212 | 161 | 0.0683 | +0.000000 |
| `no_source_voi_probability` | 0.0500 | 114 | 96 | 0.0833 | 0.1453 | `False` | 130 | 107 | 0.0935 | +0.000000 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.017986`; 
positive fraction: `0.0342`.
