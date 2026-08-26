# h17_mixed_regime_pairwise_two_stage_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0200 | 11 | 226 | 0.3938 | 0.4503 | `False` | 20 | 197 | 0.4315 | +0.000000 |
| `source_voi_probability` | 0.0200 | 12 | 225 | 0.3956 | 0.4522 | `False` | 25 | 187 | 0.4118 | +0.000000 |
| `no_source_voi_expected` | 0.0500 | 86 | 185 | 0.3459 | 0.4077 | `False` | 92 | 164 | 0.3963 | +0.000000 |
| `no_source_voi_probability` | 0.1000 | 214 | 96 | 0.0000 | 0.0307 | `True` | 187 | 94 | 0.0000 | +0.001084 |
| `external_variance` | 0.0200 | 0 | 228 | 0.4079 | 0.4643 | `False` | 0 | 197 | 0.4467 | +0.000000 |

Feedback realized VoE mean: `-0.015730`; 
positive fraction: `0.0677`.
