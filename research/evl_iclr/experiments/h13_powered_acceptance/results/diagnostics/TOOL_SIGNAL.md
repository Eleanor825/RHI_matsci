# Real Scientific Tool Signal Diagnostic

Post-hoc diagnostic on consumed folds; it is not confirmatory evidence.
Tool replicas and train-only tool models are frozen before each test fold.

| fold | task | worthy rate | AUROC | AP | gain corr. | ext-var/error Spearman | best positive prefix | prefix precision | prefix population gain |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | `matbench_pairwise` | 0.0883 | 0.9536133193048087 | 0.7736734219306357 | +0.8727 | +0.4006 | `top_0.050` | 0.8875 | +0.002341 |
| 0 | `discover_unique` | 0.1044 | 0.7323881753330084 | 0.35258672463702034 | +0.2256 | -0.1316 | `none` | 0.0000 | +0.000000 |
| 0 | `extreme_properties` | 0.0400 | 0.50390625 | 0.04030226700251889 | +0.0307 | +0.0000 | `top_0.050` | 0.8000 | +0.001273 |
