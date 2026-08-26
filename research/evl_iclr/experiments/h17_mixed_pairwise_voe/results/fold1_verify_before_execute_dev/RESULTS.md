# h17_verify_before_execute_development

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 92 | 62 | 0.0968 | 0.1821 | `True` | 72 | 57 | 0.0702 | +0.000808 |
| `source_voi_probability` | 0.0500 | 93 | 62 | 0.0968 | 0.1821 | `True` | 72 | 57 | 0.0702 | +0.000808 |
| `no_source_voi_expected` | 0.0500 | 100 | 67 | 0.0896 | 0.1691 | `True` | 92 | 71 | 0.1268 | +0.000768 |
| `no_source_voi_probability` | 0.0500 | 82 | 57 | 0.0877 | 0.1756 | `True` | 70 | 56 | 0.1429 | +0.000688 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016717`; 
positive fraction: `0.0606`.
