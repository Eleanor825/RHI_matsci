# h19_phonons_frozen_external_validation

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 61 | 33 | 0.1212 | 0.2563 | `False` | 65 | 25 | 0.1200 | +0.000000 |
| `source_voi_probability` | 0.1000 | 59 | 25 | 0.1200 | 0.2817 | `False` | 82 | 27 | 0.1111 | +0.000000 |
| `no_source_voi_expected` | 0.1000 | 59 | 33 | 0.1212 | 0.2563 | `False` | 55 | 21 | 0.0476 | +0.000000 |
| `no_source_voi_probability` | 0.1000 | 52 | 29 | 0.1034 | 0.2461 | `False` | 47 | 19 | 0.0526 | +0.000000 |
| `source_voi_tail` | 0.1000 | 52 | 22 | 0.1364 | 0.3159 | `False` | 89 | 28 | 0.1071 | +0.000000 |
| `no_source_voi_tail` | 0.1000 | 56 | 31 | 0.1290 | 0.2714 | `False` | 48 | 18 | 0.0556 | +0.000000 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.014881`; 
positive fraction: `0.0771`.
