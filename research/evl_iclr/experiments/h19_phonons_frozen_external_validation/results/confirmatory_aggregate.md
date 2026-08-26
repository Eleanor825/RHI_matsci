# H19 Frozen External Validation Aggregate

All five folds were run after the protocol lock and none were used for method selection.

| method | certified folds | mean deployed test gain | fold gains |
|---|---:|---:|---|
| `source_voi_expected` | 3/5 | +0.001406 | +0.001718, +0.000000, +0.002492, +0.002820, +0.000000 |
| `source_voi_probability` | 3/5 | +0.001409 | +0.001591, +0.000000, +0.002555, +0.002900, +0.000000 |
| `no_source_voi_expected` | 3/5 | +0.001470 | +0.001773, +0.000000, +0.002514, +0.003064, +0.000000 |
| `no_source_voi_probability` | 3/5 | +0.001459 | +0.001712, +0.000000, +0.002504, +0.003080, +0.000000 |
| `source_voi_tail` | 2/5 | +0.000913 | +0.001671, +0.000000, +0.000000, +0.002896, +0.000000 |
| `no_source_voi_tail` | 3/5 | +0.001444 | +0.001695, +0.000000, +0.002530, +0.002997, +0.000000 |
| `external_variance` | 1/5 | +0.000490 | +0.000000, +0.000000, +0.000000, +0.002450, +0.000000 |

## Frozen success criteria

- `certifies_at_least_four_of_five`: `False`
- `positive_mean_gain`: `True`
- `beats_no_source_expected`: `False`
- `beats_static_external_variance`: `True`

Overall H19 confirmatory pass: `False`.

The primary produced positive risk-controlled gain and beat static variance, but it certified on only three folds and did not beat the matched no-source estimator. H19 therefore falsifies the current claim that external-source decomposition reliably improves aggregate acquisition utility on a new property task.
