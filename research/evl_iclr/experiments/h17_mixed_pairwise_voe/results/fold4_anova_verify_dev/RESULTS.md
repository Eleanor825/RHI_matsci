# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 133 | 118 | 0.1186 | 0.1792 | `True` | 153 | 133 | 0.0977 | +0.001736 |
| `source_voi_probability` | 0.0500 | 154 | 132 | 0.1288 | 0.1869 | `True` | 171 | 144 | 0.1042 | +0.001774 |
| `no_source_voi_expected` | 0.0500 | 132 | 117 | 0.1197 | 0.1807 | `True` | 155 | 135 | 0.0963 | +0.001794 |
| `no_source_voi_probability` | 0.0500 | 133 | 118 | 0.1186 | 0.1792 | `True` | 151 | 133 | 0.0902 | +0.001801 |
| `source_voi_tail` | 0.0500 | 172 | 140 | 0.1357 | 0.1927 | `True` | 175 | 144 | 0.1111 | +0.001659 |
| `no_source_voi_tail` | 0.0500 | 135 | 119 | 0.1176 | 0.1778 | `True` | 156 | 136 | 0.0956 | +0.001749 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016421`; 
positive fraction: `0.0580`.
