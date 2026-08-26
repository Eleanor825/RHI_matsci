# H17 Numeric Multi-Tool VoE Development

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001001 | `{"high_fidelity": 107}` | `{"high_fidelity": 73}` |
| `no_source_multitool_expected` | `True` | +0.001019 | `{"high_fidelity": 110}` | `{"high_fidelity": 74}` |
| `fixed_high_fidelity_expected` | `True` | +0.001001 | `{"high_fidelity": 107}` | `{"high_fidelity": 73}` |
