# H18 Dielectric Confirmatory Aggregate

Folds 1-4 were run after protocol lock `9de82c...e5cd`.

| method | certified folds | mean deployed test gain | fold gains |
|---|---:|---:|---|
| `source_voi_expected` | 4/4 | +0.001192 | +0.000440, +0.000704, +0.001098, +0.002525 |
| `source_voi_probability` | 4/4 | +0.001209 | +0.000499, +0.000710, +0.001156, +0.002471 |
| `no_source_voi_expected` | 4/4 | +0.001172 | +0.000537, +0.000559, +0.001124, +0.002466 |
| `no_source_voi_probability` | 4/4 | +0.001212 | +0.000578, +0.000673, +0.001116, +0.002482 |
| `external_variance` | 0/4 | +0.000000 | +0.000000, +0.000000, +0.000000, +0.000000 |

- Preregistered primary pass: `False`.
- Primary minus matched no-source mean gain: `-0.00000326`.
- The verify-before-execute primary certified all four folds and beat static external variance, but did not beat its matched no-source VoE ablation.
