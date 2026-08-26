# H11.1 Five-Seed Safe-Activation Replay

| seed | policy | provenance | coverage | risk | population net gain |
|---:|---|---|---:|---:|---:|
| 1 | `no_op` | `deterministic_zero_certificate_replay` | 0.0000 | 0.0000 | +0.000000 |
| 7 | `activate_frozen_contract` | `direct_h11_1_rerun` | 0.0151 | 0.0851 | +0.000454 |
| 13 | `no_op` | `deterministic_zero_certificate_replay` | 0.0000 | 0.0000 | +0.000000 |
| 21 | `no_op` | `deterministic_zero_certificate_replay` | 0.0000 | 0.0000 | +0.000000 |
| 42 | `no_op` | `deterministic_zero_certificate_replay` | 0.0000 | 0.0000 | +0.000000 |

## Aggregate

- Active seeds: `1/5`.
- Nonnegative-gain seeds: `5/5`.
- Positive-gain seeds: `1/5`.
- Nonzero-coverage seeds: `1/5`.
- Risk at most 0.20: `5/5`.
- Mean population net gain: `+0.000091`.

H11.1 removes negative tool cost in zero-certificate seeds but does not solve the underlying generalization problem: only seed 7 activates and obtains positive gain.
