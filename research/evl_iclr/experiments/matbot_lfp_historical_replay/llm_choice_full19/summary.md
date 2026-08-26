# LFP Historical Action-Choice Pilot

- Records: 19
- Replicas per model: 3
- Selection is outcome-blind and development-only.

| model | accuracy | Brier | ECE | flip fraction | choice entropy | self uncertainty | corr(self, entropy) | corr(entropy, error) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 0.737 | 0.185 | 0.174 | 0.105 | 0.097 | 0.338 | 0.474 | 0.574 |
| gpt-5.6-luna | 0.684 | 0.197 | 0.245 | 0.211 | 0.193 | 0.335 | 0.351 | 0.482 |
| gpt-5.6-terra | 0.842 | 0.193 | 0.231 | 0.263 | 0.242 | 0.357 | 0.304 | -0.259 |
| all_models | 0.789 | 0.190 | 0.234 | 0.421 | 0.325 | 0.343 | 0.690 | 0.222 |

Self-reported uncertainty is not treated as internal variance. The empirical choice distribution across independent calls is the measured internal signal.
