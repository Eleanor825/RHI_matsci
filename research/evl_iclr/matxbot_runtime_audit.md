# MatXBot Runtime Trajectory Audit

Date: 2026-08-22

Audited local repository: `/Users/huan/matxbot`.

## Existing runtime structure

- The orchestrator exposes explicit stages: intake, research, generation,
  merge/dedupe, screening, selection, grouping, deposition, and reporting.
- Every orchestrated run is expected to emit a run ID, system, stage, context
  bundle, dispatched skills, generated files, and next checkpoint.
- The stage-state schema provides `run_id`, `system`, `stage`, `status`, and
  `updated_at`, which can anchor complete scientific trajectories.

## Missing logging required by SD-Worth

- OpenClaw HTTP hooks are disabled.
- The `session-logs` skill is disabled.
- The configured text log is an operational gateway log, not a structured
  action/outcome dataset.
- No current artifact records the proposed scientific action, numerical
  internal readouts, external tool replicas, delayed outcome, action cost, and
  parent trajectory in one schema.

## Reusable integration boundary

The MatXBot stage transition is the correct trajectory unit. Before each
expensive stage transition, the host should emit a pre-action event containing:

```text
run_id, parent_step_id, system, stage, visible_context, proposed_action,
available_tools, model_replica_readouts, evidence_replica_outputs,
estimated_cost, reversibility
```

After tool or laboratory completion, a separate outcome event should append:

```text
tool_result, execution_cost, observed_success, scientific_utility,
failure_harm, next_stage, timestamp
```

Hidden outcomes must never be copied into the pre-action event. The harness
returns numerical gain-distribution fields only; MatXBot retains routing
authority.

## Current conclusion

The repository is integration-ready at the stage-contract level but does not
yet provide real MatXBot trajectories. Enabling structured event export and
running actual stage transitions remain required before claiming online or
runtime evidence.
