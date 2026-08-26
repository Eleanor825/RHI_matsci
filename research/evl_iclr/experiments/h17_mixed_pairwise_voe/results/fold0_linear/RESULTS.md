# H15 Powered Unique Two-Stage VoE

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 93 | 71 | 0.0423 | 0.1056 | `True` | 108 | 83 | 0.0723 | +0.000912 |
| `source_voi_probability` | 0.0100 | 7 | 95 | 0.3053 | 0.3921 | `False` | 12 | 106 | 0.3113 | +0.000000 |
| `no_source_voi_expected` | 0.0100 | 36 | 90 | 0.2778 | 0.3658 | `False` | 37 | 101 | 0.2772 | +0.000000 |
| `no_source_voi_probability` | 0.0500 | 92 | 67 | 0.0299 | 0.0910 | `True` | 105 | 78 | 0.0641 | +0.000882 |
| `external_variance` | 0.0500 | 121 | 86 | 0.0814 | 0.1474 | `True` | 129 | 98 | 0.1122 | +0.000832 |

Feedback realized VoE mean: `-0.015961`; 
positive fraction: `0.0683`.
