# H17 Confirmatory Aggregate

Folds 1-4 were run after the fold-0 protocol lock.

| method | certified folds | mean deployed test gain | fold gains |
|---|---:|---:|---|
| `source_voi_expected` | 2/4 | +0.000552 | +0.000549, +0.001660, +0.000000, +0.000000 |
| `source_voi_probability` | 3/4 | +0.001207 | +0.000808, +0.002038, +0.000000, +0.001981 |
| `no_source_voi_expected` | 2/4 | +0.000615 | +0.000700, +0.000000, +0.000000, +0.001759 |
| `no_source_voi_probability` | 3/4 | +0.000890 | +0.000666, +0.000000, +0.001084, +0.001811 |
| `external_variance` | 0/4 | +0.000000 | +0.000000, +0.000000, +0.000000, +0.000000 |

- Preregistered primary pass: `False`.
- The expected-VoE primary did not satisfy the four-fold gate.
- The secondary source-aware probability score certified 3/4 folds and improved mean deployed gain over its matched no-source score by 35.5%.
- Static external variance certified 0/4 folds.
