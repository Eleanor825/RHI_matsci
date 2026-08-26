# H17 Numeric Multi-Tool VoE Development

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `False` | +0.000000 | `{"composition": 0, "high_fidelity": 0, "structure": 107}` | `{"composition": 0, "high_fidelity": 0, "structure": 40}` |
| `no_source_multitool_expected` | `False` | +0.000000 | `{"composition": 199, "high_fidelity": 0, "structure": 0}` | `{"composition": 116, "high_fidelity": 0, "structure": 0}` |
| `fixed_high_fidelity_expected` | `True` | +0.001001 | `{"high_fidelity": 107}` | `{"high_fidelity": 73}` |
