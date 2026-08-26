# H17 Numeric Multi-Tool VoE Development

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001860 | `{"high_fidelity": 229}` | `{"high_fidelity": 180}` |
| `no_source_multitool_expected` | `True` | +0.001730 | `{"high_fidelity": 164}` | `{"high_fidelity": 137}` |
| `fixed_high_fidelity_expected` | `True` | +0.001860 | `{"high_fidelity": 229}` | `{"high_fidelity": 180}` |
