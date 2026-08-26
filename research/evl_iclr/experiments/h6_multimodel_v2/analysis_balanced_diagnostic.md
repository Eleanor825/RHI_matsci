# H6.1 Corrected Balanced Diagnostic

The corrected run completed all `54/54` cells and contains exactly three
worthy and three unworthy actions, one of each per task.

## Results

- Raw three-model ensemble Brier: `0.3681`.
- Raw ensemble ECE: `0.4919`.
- Raw ensemble AURC: `0.5028`.
- Raw threshold-0.5 accuracy: `0.50`.
- Expected-gain MAE: `0.1870`.
- Mean internal variance: `0.002294`.
- Mean external variance: `0.009775`.
- Correlation of internal variance with absolute gain error: `0.8667`.
- Correlation of external variance with absolute gain error: `0.6766`.

Among the three models, Luna and Terra have lower Brier/AURC than Sol, but all
raw probabilities are poorly calibrated. The largest external variance and
gain error occur on the unique-material task; extreme-property predictions
have much smaller source variance and error.

## Interpretation

This exploratory result supports the source-decomposition mechanism but not a
direct-judge baseline claim. Real closed-model replicas produce source
variances that track gain error strongly, while their uncalibrated action-
worthiness probabilities are unreliable. This is consistent with SD-Worth's
design: use model/evidence replicas as uncertainty inputs to a task-conditional
gain calibrator rather than trusting raw LLM confidence.

The sample is label-stratified using evaluation-only worthiness and contains
only six actions. It cannot estimate natural test performance or replace the
locked 30-action complete-matrix pilot.

