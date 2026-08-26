# h20_perovskite_multitool_external_validation

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001735 | `{"high_fidelity": 201}` | `{"high_fidelity": 156}` |
| `no_source_multitool_expected` | `True` | +0.001871 | `{"high_fidelity": 255}` | `{"high_fidelity": 186}` |
| `fixed_high_fidelity_expected` | `True` | +0.001735 | `{"high_fidelity": 201}` | `{"high_fidelity": 156}` |
| `static_reliability_high_fidelity` | `False` | +0.000000 | `{}` | `{}` |
