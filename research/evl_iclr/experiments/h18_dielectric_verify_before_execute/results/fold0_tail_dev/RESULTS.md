# h18_dielectric_verify_before_execute_voe

Fold 0 development only; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.0500 | 154 | 104 | 0.0962 | 0.1576 | `True` | 117 | 83 | 0.1084 | +0.000391 |
| `source_voi_probability` | 0.0500 | 150 | 103 | 0.0971 | 0.1591 | `True` | 114 | 80 | 0.0875 | +0.000400 |
| `no_source_voi_expected` | 0.0500 | 152 | 102 | 0.0882 | 0.1489 | `True` | 112 | 80 | 0.1000 | +0.000384 |
| `no_source_voi_probability` | 0.0500 | 148 | 101 | 0.0891 | 0.1503 | `True` | 105 | 75 | 0.1067 | +0.000350 |
| `source_voi_tail` | 0.0500 | 133 | 96 | 0.0938 | 0.1579 | `True` | 97 | 67 | 0.0896 | +0.000286 |
| `no_source_voi_tail` | 0.0500 | 150 | 103 | 0.0874 | 0.1475 | `True` | 105 | 75 | 0.1067 | +0.000355 |
| `external_variance` | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |

Feedback realized VoE mean: `-0.016597`; 
positive fraction: `0.0500`.
