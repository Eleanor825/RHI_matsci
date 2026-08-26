# h20_perovskite_multitool_external_validation

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001768 | `{"high_fidelity": 371}` | `{"high_fidelity": 240}` |
| `no_source_multitool_expected` | `True` | +0.001767 | `{"high_fidelity": 376}` | `{"high_fidelity": 243}` |
| `fixed_high_fidelity_expected` | `True` | +0.001768 | `{"high_fidelity": 371}` | `{"high_fidelity": 240}` |
| `static_reliability_high_fidelity` | `False` | +0.000000 | `{}` | `{}` |
