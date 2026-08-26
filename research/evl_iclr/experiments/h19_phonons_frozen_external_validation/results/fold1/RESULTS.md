# h19_phonons_frozen_external_validation

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0200 | 0 | 0 | 0.0000 | 0.0000 | `False` | 23 | 19 | 0.0526 | +0.000000 |
| `source_voi_probability` | 0.0200 | 8 | 3 | 0.0000 | 0.6316 | `False` | 25 | 16 | 0.0625 | +0.000000 |
| `no_source_voi_expected` | 0.0200 | 3 | 3 | 0.0000 | 0.6316 | `False` | 29 | 20 | 0.0500 | +0.000000 |
| `no_source_voi_probability` | 0.0200 | 2 | 2 | 0.0000 | 0.7764 | `False` | 27 | 20 | 0.0500 | +0.000000 |
| `source_voi_tail` | 0.0500 | 22 | 11 | 0.2727 | 0.5644 | `False` | 90 | 35 | 0.1429 | +0.000000 |
| `no_source_voi_tail` | 0.0200 | 5 | 4 | 0.2500 | 0.7514 | `False` | 31 | 23 | 0.0000 | +0.000000 |
| `external_variance` | 0.0100 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.018001`; 
positive fraction: `0.0257`.
