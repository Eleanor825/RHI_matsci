# LFP Historical Action-Choice Pilot

- Records: 310
- Replicas per model: 1
- Selection is outcome-blind and development-only.

| model | accuracy | Brier | ECE | flip fraction | choice entropy | self uncertainty | corr(self, entropy) | corr(entropy, error) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 0.584 | 0.239 | 0.066 | 0.000 | 0.000 | 0.378 | 0.000 | 0.000 |
| gpt-5.6-luna | 0.603 | 0.236 | 0.021 | 0.000 | 0.000 | 0.390 | 0.000 | 0.000 |
| gpt-5.6-terra | 0.548 | 0.250 | 0.062 | 0.000 | 0.000 | 0.404 | 0.000 | 0.000 |
| all_models | 0.568 | 0.237 | 0.041 | 0.452 | 0.415 | 0.391 | 0.182 | 0.190 |

Self-reported uncertainty is not treated as internal variance. The empirical choice distribution across independent calls is the measured internal signal.
