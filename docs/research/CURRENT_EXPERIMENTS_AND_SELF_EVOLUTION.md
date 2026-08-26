# Current Experiments and Signal-Aware Self-Evolution

## 1. Current repository scope

The repository currently contains four layers of experiments:

1. **Data and label audits**: leakage checks, label consistency, utility
   recomputation, and split/group audits for the three materials tasks.
2. **Static uncertainty baselines**: verbal confidence, evidence heuristic,
   reliability-only gate, full reliability gate, self-consistency and semantic
   entropy proxies, ensemble uncertainty, cost-only policies, and utility/VoI
   acquisition policies.
3. **Recursive harness experiments**: original reliability-only RHI,
   Sci-VoI-RHI, acceptance-policy ablations, component ablations, and the
   earlier H0--H3 evolution diagnostic.
4. **LLM judge comparisons**: cached one-shot `gpt-5.5` direct judge and a
   hybrid local-VoI plus LLM judge subset. These are separate from the main
   non-LLM held-out-regime result because the full current-materials LLM run is
   not complete.

## 2. Main materials benchmark

The current main benchmark has 20,987 action records from 45 held-out regimes:

| Task | Records | Regimes | Action meaning |
| --- | ---: | ---: | --- |
| `matbench_pairwise` | 8,000 | 28 | Select the better candidate under noisy surrogate ordering |
| `discover_unique` | 10,987 | 7 | Screen a high-performance and chemically unique candidate |
| `extreme_properties` | 2,000 | 10 | Advance a candidate expected to satisfy extreme-property bounds |

The packaged data are in
`benchmarks/materials_action_trajectories_v1/`. There are two equivalent views:

- `actions.jsonl.gz`: one complete action-level decision unit per line.
- `trajectories.jsonl.gz`: 45 grouped benchmark-regime trajectories whose
  ordered actions remain independently supervised.

The action-level unit is the correct training unit for uncertainty. The grouped
view is used to preserve regime context, avoid random leakage across related
candidates, and support future MatBot trajectory replay. These are complete
offline reconstructions, not online MatBot logs.

## 3. Current non-LLM results

The complete main-materials Sci-VoI-RHI run is
`runs/scivoi_rhi_main_materials_v1/README.md`.

| Method | Net scientific utility | Risk-adjusted utility | Selective risk | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| `scivoi_policy_always_accept` | 0.7655 ± 0.2041 | 0.7233 | 0.1688 | 0.5496 |
| `scivoi_rhi` | 0.7633 ± 0.1923 | 0.7096 | 0.2148 | 0.5459 |
| `original_rhi` | 0.7282 ± 0.2020 | 0.6471 | 0.3244 | 0.5459 |
| `static_full_reliability` | 0.6421 | 0.5546 | 0.3498 | 0.4630 |
| `static_voi` | 0.7657 | 0.6869 | 0.3149 | 0.5496 |
| `self_consistency_proxy` | 0.8370 | 0.7406 | 0.3858 | 0.5778 |

The current result supports a **risk-adjusted** claim rather than a raw utility
claim: self-consistency can obtain higher raw utility while taking much more
risk, whereas the Sci-VoI variants achieve a better utility--risk balance.
The run includes 225 held-out-regime/seed folds and should be treated as the
main offline result.

## 4. LLM judge results currently available

The complete cached historical subset is
`runs/direct_judge_subset500_v1/README.md`:

- 500 records, balanced across three auxiliary task families.
- Direct `gpt-5.5` judge: score `0.3602`, Risk@10% `0.3462`, hit rate `0.6800`,
  ECE `0.1548`.
- It is a one-shot judge, not a recursive harness and not a full current-main
  materials evaluation.

The current-materials `gpt-5.5` cache is only 474/500 records, so it is marked
partial in `runs/direct_judge_main_materials_gpt55_partial_v1/`. The current
provider route does not expose a usable `gpt-5.5` channel. It should not be
used as a complete main-materials comparison.

## 5. Prompt-conditioned trajectories

The first trajectory supplement is in `runs/prompt_trajectory_main_v1/`:

- 300 sampled instances per task, 900 trajectories total;
- at most three steps per trajectory;
- task-specific system prompt and state-dependent user prompt;
- action menu containing execute, retrieve, simulate, ask-expert, and abstain;
- internal/external signals recorded at every step;
- hidden outcomes retained only for evaluation.

This first run is **prompt-conditioned but not LLM-generated**:
`agent_backend=deterministic_prompt_policy` and `llm_generated=false`. The
purpose is to validate the trajectory schema, multi-step routing, and leakage
contract without introducing provider variance. The next controlled comparison
should replace only the agent backend with a fixed LLM while preserving the
same sampled instances, prompts, action menu, and hidden outcomes.

The fixed-policy baseline comparison is in
`runs/prompt_trajectory_main_v1/baselines/BASELINE_RESULTS.md`:

| Method | Coverage | Selective risk | Hit rate | Net utility/action | Net utility/trajectory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `always_execute` | 1.000 | 0.941 | 0.059 | -0.0783 | -0.0783 |
| `random_top10` | 0.100 | 0.978 | 0.022 | -0.0935 | -0.0093 |
| `verbal_confidence_top10` | 0.100 | 0.967 | 0.033 | -0.0907 | -0.0091 |
| `evidence_heuristic_top10` | 0.100 | 0.700 | 0.300 | -0.0094 | -0.0009 |
| `static_reliability_top10` | 0.100 | 0.656 | 0.344 | -0.0074 | -0.0007 |
| `prompt_policy` | 0.367 | 0.839 | 0.161 | -0.0083 | -0.0031 |
| `oracle_top10_reference` | 0.100 | 0.422 | 0.578 | 0.1253 | 0.0125 |

The prompt policy is only a deterministic construction baseline, not our
proposed Sci-VoI-RHI. Direct LLM judge and Sci-VoI-RHI trajectory replay are
still pending and must be run on this exact fixed trajectory set.

The first trajectory-level Sci-VoI replay is now in
`runs/prompt_trajectory_main_v1/scivoi/scivoi_results.json`, with the combined
comparison in `runs/prompt_trajectory_main_v1/COMBINED_RESULTS.md`. Static
Sci-VoI and Sci-VoI-RHI were trained on the remaining historical records after
removing the 900 sampled trajectory records. On runtime routes, static Sci-VoI
selects 6.4% of actions at risk `0.914`, while Sci-VoI-RHI selects 4.7% at risk
`0.857`. These are diagnostic first results: risk is still too high, and the
trajectory protocol needs task-balanced training/calibration before supporting
a positive method claim.

### Combined action and trajectory feedback RHI

The combined-feedback experiment is in
`runs/prompt_trajectory_main_v1/combined_rhi/trajectory_combined_rhi_report.json`
and uses the same 900 complete trajectories with a disjoint
`538/124/88/150` train/feedback/acceptance/test split. The proposer receives
both feedback granularities:

- action-level: confident errors, calibration error, evidence conflict, OOD,
  cost, and over-abstention;
- trajectory-level: wrong final commitment, missed good opportunity, failed
  recovery, costly failure, over-verification, and unnecessary abstention.

The loop still makes **one harness proposal per RHI round**, not one proposal
per action. The candidate is schema-validated and compared with its
predecessor on a fresh acceptance shard. The held-out test is used only after
selection.

On the fixed top-10% test budget, H0 reaches coverage `0.100`, risk `0.867`,
success `0.133`, and net utility per trajectory `-0.0011`. The first combined
candidate H1 is accepted by the acceptance shard, but its test result degrades
to risk `0.933`, success `0.067`, and net utility per trajectory `-0.0043`.
The next two candidates are rejected, so the active harness remains H1. This
is a useful negative/diagnostic result: the combined feedback path is
implemented and acceptance prevents further regressions, but this deterministic
prompt-trajectory benchmark does **not** yet support a positive claim that
trajectory outcome feedback improves discovery utility. Runtime-route
coverage is zero on this split because the learned selective-risk threshold is
too conservative; fixed-budget results are therefore the primary comparison.

## 6. Signal-aware self-evolution

The new training loop is:

```text
H_i
  -> train action-level gate on train split
  -> replay feedback trajectories
  -> estimate internal/external signal error associations
  -> mutate prompt/contract/features/hops
  -> retrain candidate gate
  -> compare on a fresh acceptance shard
  -> accept or retain predecessor
  -> final evaluation on untouched test regimes
```

### Internal signals

These describe the agent's own epistemic state:

- verbal confidence;
- perturbation stability;
- self-consistency or semantic-entropy proxy;
- model disagreement;
- calibrated probability.
- answer-logit difference, execute probability, calibrated answer-logit
  confidence, and answer-logit uncertainty.

### External signals

These describe evidence and the scientific environment:

- evidence support and conflict;
- source reliability or source risk;
- tool agreement;
- OOD score;
- action cost and reversibility;
- context and candidate-structure descriptors.

At each round, the proposer computes a feedback-split absolute association
between each observed signal and the current execution error. Signals with
sufficient association are promoted into the next harness contract. This is not
an oracle feature-selection step: test labels are not visible to the proposer.
The candidate harness remains schema-validated, and acceptance is still based
on held-out acceptance records.

Both `src/harness_matsci/rhi.py` and `src/harness_matsci/voi.py` implement this
logic. The Sci-VoI path additionally uses the promoted signals in the utility,
uncertainty, and routing heads, so the mutation changes an executable decision
policy rather than only changing prose.

The answer-logit fields follow the decision-time internal uncertainty idea in
arXiv:2607.12447: the prompt ends before a binary `A` (execute) / `B` (defer)
decision, and the next-token logit difference is converted to calibrated
confidence and uncertainty. This requires a model endpoint that exposes token
logits; verbal confidence is not a substitute for this signal.

The current historical benchmark does not yet contain the full contract. The
coverage audit in `runs/signal_contract_audit_v1/signal_coverage.json` shows
that only `verbal_confidence` and `perturbation_stability` have complete
internal coverage; `tool_agreement` and all answer-logit fields are absent.
Therefore all results produced before full-signal enrichment must be described
as partial-signal experiments. New full-signal experiments should enrich the
records and enable `strict_signal_contract`, which rejects training whenever a
required signal is missing instead of silently filling it with zero.

## 7. What is still missing

- A full current-materials direct-judge comparison with the same 20,987-record,
  45-regime protocol.
- Multiple external LLM judge families under a fixed prompt and cost budget.
- Real MatBot runtime trajectories with action timestamps, tool calls, fallback
  routes, and post-action scientific outcomes.
- A causal online evaluation showing that `retrieve_more`, `simulate`, or
  `ask_expert` actually improves downstream discovery rather than only changing
  the offline gate score.

Until those are added, the strongest defensible statement is: the repository
has a complete offline action-level benchmark and a reproducible
signal-aware Sci-VoI-RHI training path, with strong risk-adjusted performance
on the current materials proxy data, but not yet a verified online MatBot
improvement.
