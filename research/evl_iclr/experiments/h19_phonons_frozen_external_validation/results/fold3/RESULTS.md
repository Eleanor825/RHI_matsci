# h19_phonons_frozen_external_validation

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 134 | 89 | 0.0225 | 0.0691 | `True` | 169 | 117 | 0.0427 | +0.002820 |
| `source_voi_probability` | 0.0500 | 109 | 78 | 0.0256 | 0.0785 | `True` | 152 | 109 | 0.0459 | +0.002900 |
| `no_source_voi_expected` | 0.0500 | 97 | 70 | 0.0286 | 0.0872 | `True` | 129 | 98 | 0.0204 | +0.003064 |
| `no_source_voi_probability` | 0.0500 | 96 | 70 | 0.0286 | 0.0872 | `True` | 128 | 98 | 0.0204 | +0.003080 |
| `source_voi_tail` | 0.0500 | 131 | 81 | 0.0247 | 0.0757 | `True` | 175 | 119 | 0.0168 | +0.002896 |
| `no_source_voi_tail` | 0.0500 | 106 | 77 | 0.0260 | 0.0795 | `True` | 144 | 106 | 0.0283 | +0.002997 |
| `external_variance` | 0.0200 | 61 | 44 | 0.0000 | 0.0658 | `True` | 97 | 75 | 0.0133 | +0.002450 |

Feedback realized VoE mean: `-0.017990`; 
positive fraction: `0.0313`.
