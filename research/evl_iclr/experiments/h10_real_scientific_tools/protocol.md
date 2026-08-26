# H10 Protocol: Real Scientific Tool Trajectories

Date locked: 2026-08-21

## Goal

Replace semi-synthetic and generic learned evidence with reproducible domain
tools whose outputs can be recomputed from the pre-action candidate and public
task specification.

## Tool contract

### Matbench pairwise

- Parse both candidate compositions from the visible action state.
- Compute Magpie composition descriptors with Matminer.
- Fit bootstrap ExtraTrees bulk-modulus models using training-split candidate
  compositions and training-split `log10(K_VRH)` values only.
- Tool output is the predicted chosen-minus-alternative property margin.

### DiSCoVeR-style unique discovery

- Compute Magpie descriptors for the visible candidate composition.
- Predict performance using a bootstrap model fitted to training-split bulk
  modulus only.
- Compute uniqueness as distance to the training composition library in
  standardized Magpie space; no test outcome is used.
- Tool output is the predicted performance/uniqueness discovery score relative
  to the train-only advancement threshold.

### Extreme-property discovery

- Retrieve the public target profile and Table-1 error bounds as task
  specification, not as generated-candidate outcome.
- Compute MW, logP, TPSA, QED, HBA, and HBD directly from the visible generated
  SMILES with RDKit.
- Predict DRD2 with bootstrap regressors fitted only to training molecules and
  training DRD2 values.
- Tool output is the worst normalized seven-property constraint slack.

## Uncertainty decomposition

External replicas are independently bootstrapped property/DRD2 models while
holding the action fixed. Internal replicas remain independent gain-head/model
replicas while holding each tool view fixed. The full agent-by-tool matrix is
required.

## Leakage and provenance

- Test generated properties, labels, utilities, and hidden margins are never
  read to construct visible outputs.
- A corruption audit mutates every audited non-training hidden outcome and must
  leave visible observations byte-identical.
- Provenance records Matminer/Pymatgen/RDKit versions, train-ID hash, model seed,
  tool stage, and that the trajectory is offline rather than live MatBot.

## Pilot gate

Seed 7 is the locked pilot. The primary
`task_sparse_tool_margin_continuous_sd_worth` method proceeds to five seeds if
it obtains nonzero exact-risk-controlled coverage, empirical risk at most
0.20, positive selected mean gain, positive population net gain, and a passing
corruption audit. H9/H9.1/H9.2 remain negative baselines.
