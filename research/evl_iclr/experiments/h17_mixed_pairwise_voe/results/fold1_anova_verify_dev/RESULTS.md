# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 73 | 52 | 0.0962 | 0.1916 | `True` | 61 | 48 | 0.0833 | +0.000656 |
| `source_voi_probability` | 0.0500 | 80 | 57 | 0.1053 | 0.1972 | `True` | 62 | 49 | 0.0816 | +0.000686 |
| `no_source_voi_expected` | 0.0500 | 103 | 67 | 0.0896 | 0.1691 | `True` | 96 | 75 | 0.1200 | +0.000822 |
| `no_source_voi_probability` | 0.0500 | 89 | 63 | 0.0952 | 0.1793 | `True` | 82 | 63 | 0.1270 | +0.000710 |
| `source_voi_tail` | 0.0500 | 102 | 63 | 0.0952 | 0.1793 | `True` | 76 | 61 | 0.0820 | +0.000813 |
| `no_source_voi_tail` | 0.0500 | 100 | 66 | 0.0909 | 0.1716 | `True` | 89 | 69 | 0.1159 | +0.000803 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016697`; 
positive fraction: `0.0606`.
