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

## Reproduction

```bash
PYTHONPATH=src python3 -m harness_matsci.cli graph-rhi-external \
  --pairwise-root benchmarks/external_h17_pairwise \
  --unique-root benchmarks/external_h15_unique \
  --extreme-root benchmarks/external_h14 \
  --folds 5 --epochs 80 \
  --out-dir runs/graph_rhi_external_five_fold_v1
```
