# H15 Powered Unique Two-Stage VoE

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 207 | 179 | 0.0000 | 0.0166 | `True` | 222 | 177 | 0.0000 | +0.003788 |
| `source_voi_probability` | 0.0500 | 101 | 158 | 0.0316 | 0.0654 | `True` | 121 | 140 | 0.0000 | +0.003634 |
| `no_source_voi_expected` | 0.1000 | 204 | 191 | 0.0314 | 0.0611 | `True` | 218 | 189 | 0.0317 | +0.003404 |
| `no_source_voi_probability` | 0.0500 | 96 | 155 | 0.0323 | 0.0666 | `True` | 111 | 135 | 0.0000 | +0.003569 |
| `external_variance` | 0.0000 | 0 | 224 | 0.1384 | 0.1822 | `True` | 0 | 214 | 0.1075 | +0.003684 |

Feedback realized VoE mean: `-0.014070`; 
positive fraction: `0.0967`.
