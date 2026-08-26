# H17 Numeric Multi-Tool VoE Development

The estimator emits one numeric expected VoE per tool; the contract derives the tool choice.

| method | certified | test gain | selected tools on test | executed tools on test |
|---|---|---:|---|---|
| `source_multitool_expected` | `True` | +0.000808 | `{"high_fidelity": 72}` | `{"high_fidelity": 57}` |
| `no_source_multitool_expected` | `True` | +0.000776 | `{"high_fidelity": 71}` | `{"high_fidelity": 56}` |
| `fixed_high_fidelity_expected` | `True` | +0.000808 | `{"high_fidelity": 72}` | `{"high_fidelity": 57}` |
