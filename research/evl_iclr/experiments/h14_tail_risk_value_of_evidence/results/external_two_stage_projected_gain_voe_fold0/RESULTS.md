# H14 Two-Stage Tail-Risk Value-of-Evidence Pilot

Fold 0 is development-only. Feedback state and VoE predictions are
cross-fitted; acceptance certifies one frozen acquisition threshold; test
copies the policy unchanged.

| method | task | acquire target | acceptance acquired | acceptance executed | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi_expected` | `external_pairwise_dielectric` | 0.0500 | 12 | 4 | 0.6407 | `False` | 19 | 5 | 0.0000 | +0.000000 |
| `source_voi_expected` | `external_unique_jdft2d` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `source_voi_expected` | `external_extreme_phonons` | 0.0000 | 0 | 11 | 0.5430 | `False` | 0 | 13 | 0.1538 | +0.000000 |
| `source_voi_probability` | `external_pairwise_dielectric` | 0.0500 | 12 | 4 | 0.6407 | `False` | 11 | 3 | 0.0000 | +0.000000 |
| `source_voi_probability` | `external_unique_jdft2d` | 0.0500 | 1 | 0 | 0.0000 | `False` | 1 | 0 | 0.0000 | +0.000000 |
| `source_voi_probability` | `external_extreme_phonons` | 0.0200 | 25 | 16 | 0.5446 | `False` | 30 | 16 | 0.2500 | +0.000000 |
| `no_source_voi_expected` | `external_pairwise_dielectric` | 0.1000 | 28 | 6 | 0.4946 | `False` | 47 | 11 | 0.0000 | +0.000000 |
| `no_source_voi_expected` | `external_unique_jdft2d` | 0.0100 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `no_source_voi_expected` | `external_extreme_phonons` | 0.0200 | 32 | 10 | 0.3360 | `False` | 36 | 10 | 0.1000 | +0.000000 |
| `no_source_voi_probability` | `external_pairwise_dielectric` | 0.1000 | 33 | 8 | 0.4006 | `False` | 56 | 20 | 0.2000 | +0.000000 |
| `no_source_voi_probability` | `external_unique_jdft2d` | 0.0100 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `no_source_voi_probability` | `external_extreme_phonons` | 0.0200 | 28 | 14 | 0.4511 | `False` | 31 | 15 | 0.2667 | +0.000000 |
| `external_variance` | `external_pairwise_dielectric` | 0.1000 | 50 | 18 | 0.4344 | `False` | 72 | 27 | 0.2963 | +0.000000 |
| `external_variance` | `external_unique_jdft2d` | 0.0500 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `external_variance` | `external_extreme_phonons` | 0.0000 | 0 | 11 | 0.5430 | `False` | 0 | 13 | 0.1538 | +0.000000 |
| `predictive_variance` | `external_pairwise_dielectric` | 0.1000 | 32 | 12 | 0.5086 | `False` | 49 | 23 | 0.3043 | +0.000000 |
| `predictive_variance` | `external_unique_jdft2d` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `predictive_variance` | `external_extreme_phonons` | 0.0000 | 0 | 11 | 0.5430 | `False` | 0 | 13 | 0.1538 | +0.000000 |

## Feedback Counterfactual Tool Value

- `external_pairwise_dielectric`: mean `-0.009136`, positive fraction `0.1582`, range `[-0.020000, +0.074424]`.
- `external_unique_jdft2d`: mean `-0.016404`, positive fraction `0.0680`, range `[-0.020000, +0.077643]`.
- `external_extreme_phonons`: mean `-0.017026`, positive fraction `0.0476`, range `[-0.020000, +0.066731]`.
