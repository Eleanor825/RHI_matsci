# h23_superconductivity_verify_before_execute

Development result; all feedback state and VoE scores are cross-fitted.

| method | acquire target | acceptance acquired | acceptance executed | acceptance risk | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.0000 | +0.000602 |
| `source_voi_probability` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.0485 | +0.000165 |
| `no_source_voi_expected` | 0.1000 | 309 | 155 | 0.0000 | 0.0191 | `True` | 329 | 165 | 0.1576 | -0.002963 |
| `no_source_voi_probability` | 0.2000 | 618 | 309 | 0.1100 | 0.1438 | `False` | 657 | 329 | 0.3951 | +0.000000 |
| `external_variance` | 0.0500 | 155 | 62 | 0.5484 | 0.6568 | `False` | 165 | 66 | 0.2424 | +0.000000 |

Feedback realized VoE mean: `-0.013760`; 
positive fraction: `0.0984`.
