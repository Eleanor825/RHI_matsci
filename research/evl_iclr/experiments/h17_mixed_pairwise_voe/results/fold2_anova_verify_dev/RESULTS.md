# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 89 | 67 | 0.0896 | 0.1691 | `True` | 158 | 131 | 0.1374 | +0.001845 |
| `source_voi_probability` | 0.0500 | 89 | 67 | 0.0896 | 0.1691 | `True` | 159 | 132 | 0.1364 | +0.001843 |
| `no_source_voi_expected` | 0.0500 | 94 | 72 | 0.0972 | 0.1749 | `True` | 172 | 137 | 0.1387 | +0.001843 |
| `no_source_voi_probability` | 0.0500 | 98 | 73 | 0.0959 | 0.1726 | `True` | 178 | 143 | 0.1469 | +0.001906 |
| `source_voi_tail` | 0.0500 | 108 | 78 | 0.0897 | 0.1620 | `True` | 193 | 155 | 0.1484 | +0.002020 |
| `no_source_voi_tail` | 0.0500 | 118 | 83 | 0.0964 | 0.1672 | `True` | 196 | 157 | 0.1529 | +0.002020 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016689`; 
positive fraction: `0.0587`.
