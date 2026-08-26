# H21 Protocol Amendment 2: Risk-Certified Recursive Signal Evolution

Recorded on August 23, 2026 while the uniform rerun of folds 1--4 was still in
progress and before any rerun result file was inspected.

This amendment adds a secondary, predeclared self-evolution analysis. It does
not replace the frozen primary analysis in `protocol.md`.

## Candidate sequence

The harness starts with no deployable contract and evaluates this fixed
mutation sequence on each fold's acceptance set:

1. `external_variance`: static reliability/variance contract;
2. `no_source_voi_expected`: learned expected evidence value without explicit
   source-decomposition features;
3. `source_voi_expected`: add internal/external source-state features;
4. `source_voi_probability`: replace the expected-value acquisition head with
   a probability-of-positive-value head.

At each cycle, the candidate replaces the incumbent only if:

1. it executes at least one acceptance action;
2. its one-sided Clopper--Pearson risk upper bound at
   `delta=0.05/4` is at most `alpha=0.10`;
3. its acceptance population gain after evidence cost exceeds the incumbent
   by more than `1e-6`.

The Bonferroni correction covers adaptive selection among the four declared
contracts. If no candidate passes, the evolved harness is `no_op`. The test
set is read only after the final acceptance-selected contract is fixed.

## Secondary success criteria

Across folds 1--4, the evolved harness must:

1. deploy in at least three folds;
2. have positive pooled test population gain;
3. have pooled empirical test risk at most `0.10`;
4. beat static `external_variance` in pooled test gain.

The source-feature mutation is considered supported only if it is accepted in
at least one fold; otherwise the result supports adaptive pruning rather than
a universal claim that source features always help.
