# h17_verify_before_execute_development

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 88 | 67 | 0.0896 | 0.1691 | `True` | 159 | 132 | 0.1364 | +0.001849 |
| `source_voi_probability` | 0.0500 | 90 | 67 | 0.0896 | 0.1691 | `True` | 161 | 133 | 0.1353 | +0.001851 |
| `no_source_voi_expected` | 0.0500 | 125 | 89 | 0.1011 | 0.1698 | `True` | 205 | 164 | 0.1585 | +0.002083 |
| `no_source_voi_probability` | 0.0500 | 109 | 81 | 0.0988 | 0.1712 | `True` | 190 | 154 | 0.1494 | +0.002039 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016630`; 
positive fraction: `0.0587`.
