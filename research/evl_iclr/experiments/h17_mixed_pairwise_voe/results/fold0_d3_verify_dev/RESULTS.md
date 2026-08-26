# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 91 | 58 | 0.0172 | 0.0792 | `True` | 108 | 73 | 0.0274 | +0.000992 |
| `source_voi_probability` | 0.0500 | 89 | 57 | 0.0175 | 0.0805 | `True` | 107 | 73 | 0.0274 | +0.001001 |
| `no_source_voi_expected` | 0.0500 | 102 | 67 | 0.0149 | 0.0689 | `True` | 113 | 75 | 0.0267 | +0.001003 |
| `no_source_voi_probability` | 0.0500 | 92 | 61 | 0.0164 | 0.0754 | `True` | 105 | 71 | 0.0282 | +0.000978 |
| `source_voi_tail` | 0.0500 | 111 | 65 | 0.0154 | 0.0709 | `True` | 125 | 86 | 0.0349 | +0.001076 |
| `no_source_voi_tail` | 0.0500 | 85 | 58 | 0.0172 | 0.0792 | `True` | 102 | 69 | 0.0290 | +0.000949 |
| `external_variance` | 0.1000 | 200 | 100 | 0.0100 | 0.0466 | `True` | 215 | 110 | 0.0273 | +0.000738 |

Feedback realized VoE mean: `-0.015188`; 
positive fraction: `0.0851`.
