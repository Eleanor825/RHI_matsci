# h17_mixed_regime_pairwise_two_stage_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 125 | 127 | 0.1496 | 0.2117 | `False` | 146 | 140 | 0.1143 | +0.000000 |
| `source_voi_probability` | 0.0500 | 142 | 137 | 0.1314 | 0.1886 | `True` | 154 | 154 | 0.1234 | +0.001981 |
| `no_source_voi_expected` | 0.0500 | 126 | 113 | 0.1150 | 0.1767 | `True` | 149 | 131 | 0.0916 | +0.001759 |
| `no_source_voi_probability` | 0.0500 | 139 | 121 | 0.1157 | 0.1750 | `True` | 158 | 137 | 0.0949 | +0.001811 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016207`; 
positive fraction: `0.0548`.
