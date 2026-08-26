# H17 Numeric Multi-Tool VoE Development

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.001131 | `{"high_fidelity": 166}` | `{"high_fidelity": 88}` |
| `no_source_multitool_expected` | `True` | +0.001145 | `{"high_fidelity": 168}` | `{"high_fidelity": 89}` |
| `fixed_high_fidelity_expected` | `True` | +0.001131 | `{"high_fidelity": 166}` | `{"high_fidelity": 88}` |
