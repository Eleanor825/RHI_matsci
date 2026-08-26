# h17_mixed_regime_pairwise_two_stage_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 90 | 66 | 0.0909 | 0.1716 | `True` | 158 | 127 | 0.1417 | +0.001660 |
| `source_voi_probability` | 0.0500 | 94 | 71 | 0.0986 | 0.1772 | `True` | 143 | 125 | 0.0880 | +0.002038 |
| `no_source_voi_expected` | 0.0500 | 90 | 86 | 0.1744 | 0.2558 | `False` | 167 | 149 | 0.1611 | +0.000000 |
| `no_source_voi_probability` | 0.0500 | 95 | 82 | 0.1341 | 0.2123 | `False` | 176 | 147 | 0.1497 | +0.000000 |
| `external_variance` | 0.0000 | 0 | 24 | 0.2083 | 0.3891 | `False` | 0 | 51 | 0.1176 | +0.000000 |

Feedback realized VoE mean: `-0.017175`; 
positive fraction: `0.0509`.
