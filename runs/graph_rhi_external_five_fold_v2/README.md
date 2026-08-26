# Graph-RHI: branching and merge experiment

All nodes use the same five folds and fixed top-10% action budget.
H0 is fixed; H1/H2 are independent mutation branches; H3 merges accepted branches.

| Method | Risk ↓ | Hit rate ↑ | Mean utility ↑ | Coverage |
|---|---:|---:|---:|---:|
| H0 fixed | 0.0082 | 0.9918 | 0.6865 | 0.1000 |
| H1 evidence/source | 0.0112 | 0.9888 | 0.7456 | 0.1000 |
| H2 OOD/stability | 0.0146 | 0.9854 | 0.6207 | 0.1000 |
| H3 merged | 0.0112 | 0.9888 | 0.7456 | 0.1000 |

H1 acceptance rate: `1.00`.
H2 acceptance rate: `0.00`.
Any-branch acceptance rate: `1.00`.
True merge acceptance rate: `0.00`.
Active merged node rate: `0.00`.

The graph retains branch lineage and only permits a true merge when at least two independently accepted branches are available.

## Harness interpretation

Each node is a complete harness, not a feature list. Nodes inherit the shared
`agent_interface`, `decision_unit`, calibration protocol, held-out acceptance
rule, and action budget from `H0`. `H1` changes the evidence/source contract
and routing policy; `H2` changes the OOD/stability contract and routing policy.
`H3` performs a component-wise merge of accepted parent harnesses and records
conflicts explicitly. In this run only H1 passes acceptance in all folds, so a
true two-parent H3 is correctly not activated.
