# h17_verify_before_execute_development

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 96 | 61 | 0.0164 | 0.0754 | `True` | 109 | 74 | 0.0270 | +0.001017 |
| `source_voi_probability` | 0.0500 | 112 | 68 | 0.0147 | 0.0679 | `True` | 116 | 80 | 0.0375 | +0.001048 |
| `no_source_voi_expected` | 0.0500 | 100 | 66 | 0.0152 | 0.0699 | `True` | 113 | 75 | 0.0267 | +0.001003 |
| `no_source_voi_probability` | 0.0500 | 99 | 65 | 0.0154 | 0.0709 | `True` | 112 | 74 | 0.0270 | +0.001001 |
| `external_variance` | 0.0500 | 109 | 68 | 0.0147 | 0.0679 | `True` | 121 | 86 | 0.0349 | +0.001166 |

Feedback realized VoE mean: `-0.015177`; 
positive fraction: `0.0851`.
