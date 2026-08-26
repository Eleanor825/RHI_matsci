# H17 Numeric Multi-Tool VoE Development

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.000639 | `{"high_fidelity": 0, "structure": 143}` | `{"high_fidelity": 0, "structure": 32}` |
| `no_source_multitool_expected` | `False` | +0.000000 | `{}` | `{}` |
| `fixed_high_fidelity_expected` | `True` | +0.001860 | `{"high_fidelity": 229}` | `{"high_fidelity": 180}` |
