# Full Same-Evidence LLM Results

The frozen feedback set contains 600 balanced trajectories. Evaluation uses all
2,538 frozen H5 acceptance trajectories, the same visible evidence, strict gain
definition, and 10% action budget as the numerical harness.

| method | selected | errors | empirical risk | CP UCB | strict gain | certified |
|---|---:|---:|---:|---:|---:|---|
| EVL portfolio KG | 254 | 2 | 0.0079 | 0.0246 | 195.7886 | yes |
| screening GPT-5.6 ensemble | 254 | 25 | 0.0984 | 0.1348 | 153.4501 | no |
| verification GPT-5.6 ensemble | 254 | 10 | 0.0394 | 0.0659 | 141.0945 | yes |

The ensemble averages GPT-5.6 Sol, Luna, and Terra and applies task-specific
Platt calibration fit only on feedback.

| paired comparison | difference | stratified bootstrap 95% CI | one-sided sign-flip p |
|---|---:|---:|---:|
| EVL minus screening ensemble | +42.3385 | [22.1953, 62.5509] | 0.00002 |
| EVL minus verification ensemble | +54.6940 | [31.0198, 78.5772] | 0.00001 |

Both frozen superiority gates pass. The screening ensemble is the strongest LLM
by gain; the verification ensemble is the safety-qualified comparator selected
on the earlier balanced subset.
