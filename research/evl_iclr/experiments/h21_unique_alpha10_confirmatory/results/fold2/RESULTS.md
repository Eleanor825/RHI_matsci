# h15_powered_unique_two_stage_voe

Confirmatory result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 230 | 146 | 0.0548 | 0.0967 | `True` | 247 | 183 | 0.0546 | +0.001362 |
| `source_voi_probability` | 0.1000 | 256 | 156 | 0.0513 | 0.0906 | `True` | 286 | 195 | 0.0615 | +0.001081 |
| `no_source_voi_expected` | 0.1000 | 249 | 154 | 0.0519 | 0.0918 | `True` | 292 | 192 | 0.0469 | +0.001297 |
| `no_source_voi_probability` | 0.1000 | 208 | 129 | 0.0620 | 0.1091 | `False` | 242 | 166 | 0.0542 | +0.000000 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.018791`; 
positive fraction: `0.0187`.
