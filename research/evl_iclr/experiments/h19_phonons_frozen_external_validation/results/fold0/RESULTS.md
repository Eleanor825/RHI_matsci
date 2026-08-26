# h19_phonons_frozen_external_validation

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0200 | 90 | 67 | 0.0000 | 0.0437 | `True` | 61 | 55 | 0.0000 | +0.001718 |
| `source_voi_probability` | 0.0200 | 71 | 57 | 0.0000 | 0.0512 | `True` | 57 | 52 | 0.0000 | +0.001591 |
| `no_source_voi_expected` | 0.0200 | 100 | 71 | 0.0000 | 0.0413 | `True` | 66 | 59 | 0.0000 | +0.001773 |
| `no_source_voi_probability` | 0.0200 | 84 | 62 | 0.0000 | 0.0472 | `True` | 60 | 54 | 0.0000 | +0.001712 |
| `source_voi_tail` | 0.0200 | 84 | 63 | 0.0000 | 0.0464 | `True` | 64 | 56 | 0.0000 | +0.001671 |
| `no_source_voi_tail` | 0.0200 | 79 | 58 | 0.0000 | 0.0503 | `True` | 57 | 53 | 0.0000 | +0.001695 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.018374`; 
positive fraction: `0.0302`.
