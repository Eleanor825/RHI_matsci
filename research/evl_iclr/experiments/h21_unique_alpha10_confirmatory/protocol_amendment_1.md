# H21 Protocol Amendment 1: Optional Tail Baseline Isolation

Recorded on August 23, 2026 after folds 1 and 2 completed but before their
result files were inspected. Fold 3 stopped before producing a result because
the optional tail-VoI baseline had insufficient hurdle components in one
cross-fit partition.

The failure is unrelated to the frozen primary `source_voi_expected`. The
implementation is amended to permit excluding tail-only comparator heads from
the run. The primary, source ablation, external-variance baseline, data folds,
risk level, costs, model families, and confirmatory gate are unchanged.

The incomplete first run is retired in full. Folds 1--4 are rerun uniformly
with `--exclude-tail-methods`; none of the first-run numerical result files are
used in analysis or aggregation.

New implementation hash for `run_external_h15_unique_voe.py`:

`6e4b10c2dcb26419da934512aa310303f91ff02f793a8946ecf2f31cdd478c71`
