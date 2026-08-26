# Outcome-Blind Same-Evidence LLM Subset Amendment

Frozen on August 23, 2026, before computing any LLM-judge metric or inspecting
any LLM prediction against an outcome.

The original protocol requested same-evidence judges on the complete feedback
and acceptance batches. Running three models at three evidence stages over all
6,139 records would require 55,251 remote judgments. This is not necessary to
estimate the direct-judge baseline and delays the confirmatory gate without
increasing the scientific-task coverage.

The same-evidence comparison therefore uses an outcome-blind, deterministic,
task-stratified subset selected only by SHA-256 of `seed || benchmark ||
record_id`:

- feedback: 200 records from each task, 600 total;
- acceptance: every available `discover_unique` record (85), 200
  `extreme_properties` records, and 200 `matbench_pairwise` records, 485 total.

Every selected record is judged by GPT-5.6 Sol, Luna, and Terra at base,
screening, and verification evidence. Calibration is fit on feedback only;
acceptance outcomes remain evaluation-only. Existing cached calls are reused
only when their record, model, stage, prompt contract, and gain definition
match exactly. The subset rule does not inspect labels, utility, gain, or any
model output.

The LLM comparison is reported as a prespecified stratified diagnostic and is
not used to tune the numerical harness. The main non-LLM acceptance certificate
continues to use the complete 2,538-record acceptance batch.
