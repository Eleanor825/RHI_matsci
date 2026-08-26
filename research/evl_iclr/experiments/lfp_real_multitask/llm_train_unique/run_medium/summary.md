# LFP Historical Action-Choice Pilot

- Records: 240
- Replicas per model: 1
- Selection is outcome-blind and development-only.

| model | accuracy | Brier | ECE | flip fraction | choice entropy | self uncertainty | corr(self, entropy) | corr(entropy, error) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 0.600 | 0.241 | 0.026 | 0.000 | 0.000 | 0.376 | 0.000 | 0.000 |
| gpt-5.6-luna | 0.542 | 0.252 | 0.061 | 0.000 | 0.000 | 0.391 | 0.000 | 0.000 |
| gpt-5.6-terra | 0.567 | 0.247 | 0.059 | 0.000 | 0.000 | 0.411 | 0.000 | 0.000 |
| all_models | 0.588 | 0.244 | 0.063 | 0.354 | 0.325 | 0.393 | 0.195 | 0.017 |

Self-reported uncertainty is not treated as internal variance. The empirical choice distribution across independent calls is the measured internal signal.
