# ICLR Requirements Audit

Status is evidence-based as of August 23, 2026. `Complete` means an artifact and
held-out result currently exist; it does not mean the paper is guaranteed to be
accepted.

| requirement | status | authoritative evidence | current boundary |
|---|---|---|---|
| Strict net scientific gain | Complete | `method_specification.md`; H5/H23 result JSON | Utility normalization and cost/harm weights are benchmark operationalizations, not universal laboratory economics. |
| Internal/external decomposition | Complete | exact crossed identity in `method_specification.md`; decomposition code/tests | Operational source decomposition, not causal identifiability. |
| Calibrated action-worthiness | Complete | task-stage Platt and split-conformal heads; 255-test suite | Marginal calibration does not imply per-action conditional validity. |
| Selective risk guarantee | Complete with assumptions | `theory_guarantees.md`; `experiments/risk_guarantee_audit/results.json` | Average conditional selected-batch risk under frozen outcome-blind selection and independent bounded losses; not individual-action coverage. |
| Real MatBot/tool trajectories | Partial | 811 fresh real-tool trajectories; three live MatBot/OpenClaw sessions; 12 measured LFP replays | Live MatBot sessions lack post-execution outcomes; measured LFP replay is small and development-only. |
| Multi-model, multi-task, held-out | Complete for current benchmark | three GPT-5.6 variants; three core tasks; group/fingerprint-disjoint H18--H23 and live-tool test | A second provider family remains desirable but unavailable on the current router. |
| Beats static reliability | Complete | H5 gain `195.7886` vs static `36.0606`; live-tool `63.427` vs `-12.192`; H23 static does not certify | Global H5 gain concentrates on pairwise actions; task-balanced stress remains weaker. |
| Beats LLM judge | Complete for both frozen comparators | full 2,538 same-evidence set: `195.7886` vs best-gain LLM `153.4501` and safety-qualified LLM `141.0945`; paired differences `+42.3385` and `+54.6940`, both with positive bootstrap intervals and `p<=0.00002` | Second-provider replication remains unavailable. |
| Risk-safe harness evolution | Complete on H23 | predeclared fixed sequence, 3/4 deployed folds, zero errors, positive held-out gain | Only one fold retained source features; gain monotonicity is empirical, not a theorem. |

## Current paper-level decision

The method and non-LLM evidence now support a coherent ICLR submission. The two
remaining acceptance-critical weaknesses are:

1. add a larger runtime-plus-outcome MatBot study when physical or historical
   outcomes become available.
2. replicate the same-evidence judge comparison with a second provider family
   when an endpoint is available.

The first is an executable benchmark gap. The second should be stated as a
real-world validation limitation unless additional outcome-linked trajectories
can be collected before submission.
