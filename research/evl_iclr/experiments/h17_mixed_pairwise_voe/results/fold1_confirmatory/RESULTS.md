# h17_mixed_regime_pairwise_two_stage_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 62 | 44 | 0.0682 | 0.1669 | `True` | 50 | 39 | 0.0769 | +0.000549 |
| `source_voi_probability` | 0.0500 | 93 | 62 | 0.0968 | 0.1821 | `True` | 72 | 57 | 0.0702 | +0.000808 |
| `no_source_voi_expected` | 0.0500 | 91 | 64 | 0.0938 | 0.1767 | `True` | 83 | 63 | 0.1270 | +0.000700 |
| `no_source_voi_probability` | 0.0500 | 83 | 58 | 0.0862 | 0.1728 | `True` | 69 | 54 | 0.1296 | +0.000666 |
| `external_variance` | 0.0000 | 0 | 21 | 0.3333 | 0.5359 | `False` | 0 | 18 | 0.1667 | +0.000000 |

Feedback realized VoE mean: `-0.016567`; 
positive fraction: `0.0574`.
