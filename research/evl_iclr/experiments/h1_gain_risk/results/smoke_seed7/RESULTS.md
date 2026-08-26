# H1 — Strict Net Gain and Risk-Controlled Worthiness

Seeds: `[7]`; alpha: `0.1`; delta: `0.05`.

| Method | Coverage | Risk | Net gain/action | Gain efficiency | Worst-regime risk | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `binary_worthiness` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00000 ± 0.00000 | 0.0000 ± 0.0000 | 0.0000 | 0.1532 | 0.0778 |
| `sd_worth_gain_distribution` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00000 ± 0.00000 | 0.0000 ± 0.0000 | 0.0000 | 0.2257 | 0.2567 |
| `static_reliability` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00000 ± 0.00000 | 0.0000 ± 0.0000 | 0.0000 | 0.1532 | 0.0778 |

The selection threshold is chosen on the independent acceptance split using a union-adjusted one-sided exact binomial risk upper bound. Test risk is reported but never used for threshold selection.
