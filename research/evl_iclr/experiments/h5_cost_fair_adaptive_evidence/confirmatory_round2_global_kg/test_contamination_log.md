# Locked-Test Contamination Log

Date: 2026-08-23

The first row of `adaptive_evidence_h5_round2_locked_test_3375.jsonl` was
accidentally printed while inspecting the runtime schema. Its hidden outcome was
therefore exposed before the confirmatory gate completed.

Consequences:

- The entire round-2 locked-test artifact is retired from confirmatory use.
- No metric will be computed or reported from that artifact.
- A replacement test set must be sampled from previously unused scientific
  fingerprints, exported to a physically separate artifact, hashed, and kept
  outcome-blind until the complete non-LLM and same-evidence LLM gate passes.

This incident does not affect the already evaluated round-2 acceptance batch,
but it invalidates the former locked-test claim.
