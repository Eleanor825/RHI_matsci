# LFP Historical Action-Choice Pilot

- Records: 6
- Replicas per model: 2
- Selection is outcome-blind and development-only.

| model | accuracy | Brier | ECE | flip fraction | choice entropy | self uncertainty | corr(self, entropy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 0.667 | 0.194 | 0.312 | 0.167 | 0.167 | 0.334 | 0.637 |
| gpt-5.6-luna | 0.667 | 0.185 | 0.210 | 0.167 | 0.167 | 0.312 | 0.219 |
| gpt-5.6-terra | 0.833 | 0.185 | 0.297 | 0.000 | 0.000 | 0.353 | 0.000 |
| all_models | 0.667 | 0.187 | 0.204 | 0.500 | 0.153 | 0.333 | 0.567 |

Self-reported uncertainty is not treated as internal variance. The empirical choice distribution across independent calls is the measured internal signal.
