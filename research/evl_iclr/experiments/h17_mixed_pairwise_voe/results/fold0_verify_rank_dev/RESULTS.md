# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 207 | 104 | 0.0385 | 0.0859 | `True` | 220 | 110 | 0.0273 | +0.000651 |
| `source_voi_probability` | 0.1000 | 207 | 104 | 0.0481 | 0.0984 | `True` | 220 | 110 | 0.0636 | +0.000540 |
| `no_source_voi_expected` | 0.1000 | 207 | 104 | 0.0192 | 0.0593 | `True` | 220 | 110 | 0.0000 | +0.000767 |
| `no_source_voi_probability` | 0.1000 | 207 | 104 | 0.0288 | 0.0729 | `True` | 220 | 110 | 0.0000 | +0.000764 |
| `source_voi_tail` | 0.1000 | 207 | 104 | 0.0288 | 0.0729 | `True` | 220 | 110 | 0.0273 | +0.000682 |
| `no_source_voi_tail` | 0.1000 | 207 | 104 | 0.0288 | 0.0729 | `True` | 220 | 110 | 0.0000 | +0.000764 |
| `external_variance` | 0.1000 | 207 | 104 | 0.0288 | 0.0729 | `True` | 220 | 110 | 0.0091 | +0.000726 |

Feedback realized VoE mean: `-0.015188`; 
positive fraction: `0.0851`.
