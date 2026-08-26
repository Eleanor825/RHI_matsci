# h20_perovskite_multitool_external_validation

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001546 | `{"high_fidelity": 384}` | `{"high_fidelity": 256}` |
| `no_source_multitool_expected` | `True` | +0.001440 | `{"high_fidelity": 429}` | `{"high_fidelity": 274}` |
| `fixed_high_fidelity_expected` | `True` | +0.001546 | `{"high_fidelity": 384}` | `{"high_fidelity": 256}` |
| `static_reliability_high_fidelity` | `False` | +0.000000 | `{}` | `{}` |
