# h20_perovskite_multitool_external_validation

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001267 | `{"high_fidelity": 366}` | `{"high_fidelity": 203}` |
| `no_source_multitool_expected` | `True` | +0.001275 | `{"high_fidelity": 361}` | `{"high_fidelity": 202}` |
| `fixed_high_fidelity_expected` | `True` | +0.001267 | `{"high_fidelity": 366}` | `{"high_fidelity": 203}` |
| `static_reliability_high_fidelity` | `False` | +0.000000 | `{}` | `{}` |
