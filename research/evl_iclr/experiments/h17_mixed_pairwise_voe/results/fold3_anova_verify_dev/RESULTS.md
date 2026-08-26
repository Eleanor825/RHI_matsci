# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 217 | 98 | 0.0000 | 0.0301 | `True` | 194 | 97 | 0.0000 | +0.001117 |
| `source_voi_probability` | 0.1000 | 194 | 92 | 0.0000 | 0.0320 | `True` | 181 | 92 | 0.0000 | +0.001124 |
| `no_source_voi_expected` | 0.1000 | 215 | 97 | 0.0000 | 0.0304 | `True` | 189 | 91 | 0.0000 | +0.000968 |
| `no_source_voi_probability` | 0.1000 | 224 | 101 | 0.0000 | 0.0292 | `True` | 196 | 94 | 0.0000 | +0.000982 |
| `source_voi_tail` | 0.1000 | 218 | 99 | 0.0000 | 0.0298 | `True` | 206 | 99 | 0.0000 | +0.001060 |
| `no_source_voi_tail` | 0.1000 | 218 | 96 | 0.0000 | 0.0307 | `True` | 190 | 95 | 0.0000 | +0.001079 |
| `external_variance` | 0.0200 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.013348`; 
positive fraction: `0.0999`.
