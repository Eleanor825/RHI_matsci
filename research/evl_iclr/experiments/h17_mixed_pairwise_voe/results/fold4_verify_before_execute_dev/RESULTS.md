# h17_verify_before_execute_development

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 111 | 101 | 0.1089 | 0.1738 | `True` | 127 | 111 | 0.0721 | +0.001675 |
| `source_voi_probability` | 0.0500 | 151 | 128 | 0.1250 | 0.1836 | `True` | 176 | 146 | 0.1096 | +0.001743 |
| `no_source_voi_expected` | 0.0500 | 121 | 109 | 0.1193 | 0.1829 | `True` | 138 | 121 | 0.0826 | +0.001732 |
| `no_source_voi_probability` | 0.0500 | 137 | 119 | 0.1176 | 0.1778 | `True` | 157 | 137 | 0.0949 | +0.001788 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016438`; 
positive fraction: `0.0580`.
