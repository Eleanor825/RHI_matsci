# h17_mixed_regime_pairwise_two_stage_voe

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.2000 | 414 | 207 | 0.1643 | 0.2126 | `False` | 440 | 220 | 0.3409 | +0.000000 |
| `source_voi_probability` | 0.2000 | 414 | 207 | 0.1836 | 0.2336 | `False` | 440 | 220 | 0.3409 | +0.000000 |
| `no_source_voi_expected` | 0.2000 | 414 | 207 | 0.1787 | 0.2283 | `False` | 440 | 220 | 0.3455 | +0.000000 |
| `no_source_voi_probability` | 0.2000 | 414 | 207 | 0.1739 | 0.2231 | `False` | 440 | 220 | 0.3455 | +0.000000 |
| `source_voi_tail` | 0.2000 | 414 | 207 | 0.1787 | 0.2283 | `False` | 440 | 220 | 0.3409 | +0.000000 |
| `no_source_voi_tail` | 0.2000 | 414 | 207 | 0.1836 | 0.2336 | `False` | 440 | 220 | 0.3455 | +0.000000 |
| `external_variance` | 0.0500 | 104 | 42 | 0.9286 | 0.9802 | `False` | 110 | 44 | 0.9545 | +0.000000 |

Feedback realized VoE mean: `-0.013348`; 
positive fraction: `0.0999`.
