# H17 RHI Contract Evolution

The evolution benchmark consists of five consumed log-KVRH group folds. A candidate is deployable only if it passes the independent certificate on every evolution fold.

| candidate | contract | head | certified folds | mean deployed gain | accepted |
|---|---|---|---:|---:|---|
| `joint_expected` | `joint` | `source_voi_expected` | 3/5 | +0.000624 | `False` |
| `joint_probability` | `joint` | `source_voi_probability` | 3/5 | +0.000965 | `False` |
| `verify_expected` | `verify_before_execute` | `source_voi_expected` | 5/5 | +0.001309 | `True` |
| `verify_probability` | `verify_before_execute` | `source_voi_probability` | 5/5 | +0.001284 | `False` |

Selected contract: `verify_expected`.

This selection uses only the H17 evolution benchmark. H18 and subsequent datasets are external transfer evaluations and are not inputs to the evolution gate.
