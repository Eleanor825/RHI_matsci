# H14 Two-Stage Tail-Risk Value-of-Evidence Pilot

Fold 0 is development-only. Feedback state and VoE predictions are
cross-fitted; acceptance certifies one frozen acquisition threshold; test
copies the policy unchanged.

| method | task | acquire target | acceptance acquired | acceptance executed | risk UCB | deployed | test acquired | test executed | test risk | deployed test gain |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `source_voi` | `external_pairwise_dielectric` | 0.0500 | 15 | 7 | 0.9148 | `False` | 21 | 12 | 0.0833 | +0.000000 |
| `source_voi` | `external_unique_jdft2d` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `source_voi` | `external_extreme_phonons` | 0.0500 | 36 | 20 | 0.1851 | `True` | 47 | 25 | 0.0000 | +0.003149 |
| `no_source_voi` | `external_pairwise_dielectric` | 0.0500 | 15 | 5 | 0.9574 | `False` | 22 | 9 | 0.0000 | +0.000000 |
| `no_source_voi` | `external_unique_jdft2d` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `no_source_voi` | `external_extreme_phonons` | 0.0200 | 29 | 7 | 0.4428 | `False` | 31 | 8 | 0.0000 | +0.000000 |
| `external_variance` | `external_pairwise_dielectric` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `external_variance` | `external_unique_jdft2d` | 0.0500 | 33 | 3 | 0.9235 | `False` | 18 | 2 | 0.0000 | +0.000000 |
| `external_variance` | `external_extreme_phonons` | 0.0100 | 2 | 9 | 0.6263 | `False` | 5 | 8 | 0.2500 | +0.000000 |
| `predictive_variance` | `external_pairwise_dielectric` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `predictive_variance` | `external_unique_jdft2d` | 0.0000 | 0 | 0 | 0.0000 | `False` | 0 | 0 | 0.0000 | +0.000000 |
| `predictive_variance` | `external_extreme_phonons` | 0.0000 | 0 | 7 | 0.7351 | `False` | 0 | 8 | 0.2500 | +0.000000 |

## Feedback Counterfactual Tool Value

- `external_pairwise_dielectric`: mean `-0.018781`, positive fraction `0.0597`, range `[-0.349137, +0.074424]`.
- `external_unique_jdft2d`: mean `-0.016633`, positive fraction `0.0583`, range `[-0.020000, +0.077643]`.
- `external_extreme_phonons`: mean `-0.017199`, positive fraction `0.0476`, range `[-0.020000, +0.066731]`.
