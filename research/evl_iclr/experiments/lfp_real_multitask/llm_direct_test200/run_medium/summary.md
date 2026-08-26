# LFP Historical Action-Choice Pilot

- Records: 200
- Replicas per model: 1
- Selection is outcome-blind and development-only.

| model | accuracy | Brier | ECE | flip fraction | choice entropy | self uncertainty | corr(self, entropy) | corr(entropy, error) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 0.600 | 0.239 | 0.059 | 0.000 | 0.000 | 0.376 | 0.000 | 0.000 |
| gpt-5.6-luna | 0.625 | 0.234 | 0.065 | 0.000 | 0.000 | 0.383 | 0.000 | 0.000 |
| gpt-5.6-terra | 0.555 | 0.247 | 0.075 | 0.000 | 0.000 | 0.403 | 0.000 | 0.000 |
| all_models | 0.625 | 0.236 | 0.048 | 0.395 | 0.363 | 0.388 | 0.332 | 0.008 |

Self-reported uncertainty is not treated as internal variance. The empirical choice distribution across independent calls is the measured internal signal.
